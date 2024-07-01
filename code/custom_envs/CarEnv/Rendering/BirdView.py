import array
import os

import cairo
import numpy as np
from PIL import Image

from .Rendering import BitmapRenderer, draw_vehicle_proxy, draw_vehicle_state
from .ConeRenderer import ConeRenderer
from .TrackRenderer import TrackRenderer


class BirdViewRenderer:
    def __init__(self, width, height, scale=15., orient_forward=False, draw_physics=False):
        #img_path = os.path.join(os.path.abspath(__file__), os.pardir, os.pardir, "steering_wheel.png")
        img_path = "custom_envs/CarEnv/steering_wheel.png"
        #print(os.path.abspath(img_path))
        self.steering_wheel_image_cairo = cairo.ImageSurface.create_from_png(img_path)
        self.steering_wheel_image_pil = Image.open(img_path)
        self._bvr = BitmapRenderer(width, height)
        self._bvr.open()
        self.width = width
        self.height = height
        self.scale = scale
        self.orient_forward = orient_forward
        self.last_transform = None
        self.slip_lines = []
        self.draw_physics = draw_physics
        self.start_lights = None
        self.ghosts = None
        self.show_track = True
        self.show_digital_tachometer = False
        self._cone_renderer = ConeRenderer()
        self._track_renderer = TrackRenderer()

    def _render_ctx(self, env, ctx):
        from ..Problems import FreeDriveProblem
        from .Gauge import Gauge

        pose = env.vehicle_model.get_pose(env.vehicle_state)[0]
        if self.orient_forward:
            ctx.translate(self.width / 2, self.height * .8)
            ctx.rotate(-pose[2] - np.pi / 2)
        else:
            ctx.translate(self.width / 2, self.height / 2)
        ctx.scale(self.scale, self.scale)
        ctx.translate(-pose[0], -pose[1])

        if isinstance(env.problem, FreeDriveProblem) and self.show_track:
            self._track_renderer.render(ctx, env.problem.track_dict['centerline'], env.problem.track_dict['width'])

        self.render_tire_slip(ctx, env)

        cones = env.objects['cones']
        types = np.argmax(cones.data[:, 2:], axis=-1) + 1

        self._cone_renderer.render(ctx, cones.data[:, :2], types, cones.radius)

        self.render_ghosts(ctx, env)
        draw_vehicle_proxy(ctx, env)

        env.problem.render(ctx, env)

        if self.draw_physics:
            ctx.identity_matrix()
            ctx.translate(100, self.height - 100)
            draw_vehicle_state(ctx, env)

        ctx.identity_matrix()
        ctx.translate(self.width - 80, self.height - 80)
        Gauge(0, 100).draw(ctx, abs(env.vehicle_last_speed) * 3.6)
        self.render_pedals(ctx, env)
        self.render_steering_wheel(ctx, env.vehicle_state[0][6])

        if self.show_digital_tachometer:
            self.render_digital_tachometer(ctx, env)

        self.render_start_lights(ctx)

    def render(self, env):
        ctx = self._bvr.clear()
        self._render_ctx(env, ctx)
        return self._bvr.get_data()

    def render_pdf(self, env, path):
        import cairo

        surface = cairo.PDFSurface(path, self.width, self.height)

        try:
            ctx = cairo.Context(surface)
            ctx.set_source_rgb(.8, .8, .8)
            ctx.fill()
            self._render_ctx(env, ctx)
        finally:
            surface.finish()

    def reset(self):
        self.last_transform = None
        self.ghosts = None
        self._track_renderer.reset()
        self.slip_lines = []
        self._cone_renderer.reset()

    def _add_slip_line(self, t1, t2, x, y):
        vec = np.array([x, y, 1])

        p1 = (t1 @ vec)[:2]
        p2 = (t2 @ vec)[:2]
        self.slip_lines.append((p1, p2))
        self.slip_lines = self.slip_lines[-500:]  # Limit to not overwhelm

    def render_steering_wheel(self, ctx: cairo.Context, angle):
        rotated_img = self.steering_wheel_image_pil.rotate(-360*angle/np.pi)
        #s = rotated_img.tostring('raw', 'BGRA')
        #a = array.array('B', s)
        #rotated_img_cairo = cairo.ImageSurface(a, cairo.FORMAT_ARGB32, rotated_img.width, rotated_img.height)
        #rotated_img.putalpha(256)
        # Instead of saving and then deleting again, put it in a buffer and pass it directly to cairo
        #rotated_img.save("rotated_steering_wheel.png", "PNG")
        rotated_img_cairo = from_pil(rotated_img)
        #rotated_img_cairo = cairo.ImageSurface.create_from_png("rotated_steering_wheel.png")
        #rotated_img_cairo = cairo.ImageSurface.create_for_data(np.array(rotated_img), cairo.FORMAT_RGB24, rotated_img.width, rotated_img.height)
        ctx.set_source_surface(rotated_img_cairo, -130, -10)
        # os.remove("rotated_steering_wheel.png")
        ctx.paint()

    def render_tire_slip(self, ctx: cairo.Context, env):
        from .Rendering import stroke_fill

        if not hasattr(env.vehicle_model, "front_slip_"):
            # Does not appear to be correct vehicle model
            return

        transform = np.linalg.inv(env.ego_transform)

        if self.last_transform is None:
            self.last_transform = transform
            return

        # Return if no physics set yet
        if env.vehicle_model.front_slip_ is None:
            return

        front_slip = env.vehicle_model.front_slip_[0]
        rear_slip = env.vehicle_model.rear_slip_[0]

        h_wb = env.vehicle_model.wheelbase / 2
        h_w = env.collision_bb[-1]

        if front_slip:
            self._add_slip_line(transform, self.last_transform, h_wb, -h_w)
            self._add_slip_line(transform, self.last_transform, h_wb, h_w)
        if rear_slip:
            self._add_slip_line(transform, self.last_transform, -h_wb, -h_w)
            self._add_slip_line(transform, self.last_transform, -h_wb, h_w)

        self.last_transform = transform

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for p1, p2 in self.slip_lines:
            ctx.move_to(*p1)
            ctx.line_to(*p2)
        stroke_fill(ctx, (0., 0., 0.), None, 3.)

    def render_pedals(self, ctx: cairo.Context, env):
        from .Rendering import stroke_fill

        if not hasattr(env.action, 'throttle_position_'):
            return

        ctx.identity_matrix()
        ctx.translate(self.width - 200, self.height - 120)

        ctx.rectangle(30, -5, 20, 85)
        stroke_fill(ctx, (0., 0., 0.), (1., 1., 1.))
        x_offset = ctx.text_extents("throttle").width / 2
        ctx.move_to(40 - x_offset, 90)
        ctx.show_text("throttle")

        bar_size = 85 * env.action.throttle_position_
        ctx.rectangle(30, 80 - bar_size, 20, bar_size)
        stroke_fill(ctx, (0., 0., 0.), (0., 1., 0.))

        ctx.rectangle(-10, -5, 20, 85)
        stroke_fill(ctx, (0., 0., 0.), (1., 1., 1.))
        x_offset = ctx.text_extents("brake").width / 2
        ctx.move_to(0 - x_offset, 90)
        ctx.show_text("brake")

        bar_size = 85 * env.action.brake_position_
        ctx.rectangle(-10, 80 - bar_size, 20, bar_size)
        stroke_fill(ctx, (0., 0., 0.), (1., 0., 0.))

    def render_digital_tachometer(self, ctx:cairo.Context, env):
        from .Rendering import stroke_fill
        v = f"{int(env.vehicle_last_speed * 3.6)}"
        ctx.select_font_face("Latin Modern Mono", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
        advance = ctx.text_extents("0").x_advance
        height = ctx.text_extents("0").height
        ctx.set_font_size(100)
        ctx.identity_matrix()

        ctx.move_to(self.width / 2 - 250, self.height)
        ctx.line_to(self.width / 2 - 130, self.height - 120)
        ctx.line_to(self.width / 2 + 130, self.height - 120)
        ctx.line_to(self.width / 2 + 250, self.height)
        ctx.close_path()
        stroke_fill(ctx, (0., 0., 0.), (0.3, 0.3, 0.3))

        for i, k in enumerate(v.rjust(3, ' ')):
            ctx.move_to(self.width / 2 - advance * 3 / 2 + i * advance + 30, self.height - 60 + height / 2)
            ctx.text_path(k)
            stroke_fill(ctx, (0., 0., 0.), (1., .3, .3))

        if hasattr(env.vehicle_model, "front_slip_") and env.vehicle_model.front_slip_ is not None and env.vehicle_model.front_slip_[0]:
            ctx.translate(0, self.height - 60 + height / 2)
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            ctx.arc(self.width / 2 - 100, -30, 20, 0, 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(self.width / 2 - 100, -30, 10, 0, 2 * np.pi)
            ctx.new_sub_path()
            ctx.rectangle(self.width / 2 - 120, -10, 40, 10)
            stroke_fill(ctx, (0., 0., 0.), (1., .3, .3))

    def render_ghosts(self, ctx: cairo.Context, env):
        if self.ghosts is None:
            return

        for color, pose in self.ghosts:
            draw_vehicle_proxy(ctx, env, pose=pose, query_env=False, color=color)

    def render_start_lights(self, ctx: cairo.Context):
        from .Rendering import stroke_fill
        r = 20

        if self.start_lights is None:
            return

        def draw(on):
            ctx.rectangle(-1.5 * r, -4 * r, 3 * r, 8 * r)
            stroke_fill(ctx, (0., 0., 0.), (.2, .2, .2))

            ctx.arc(0., 0., r, 0., 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(0., -2.5 * r, r, 0., 2 * np.pi)
            ctx.new_sub_path()
            ctx.arc(0., 2.5 * r, r, 0., 2 * np.pi)
            stroke_fill(ctx, (0., 0., 0.), (1., 0., 0.) if on else (.4, .2, .2))

        ctx.identity_matrix()
        ctx.translate(self.width * .5 - 3.5 * r, self.height * .2)
        draw(self.start_lights >= 1)
        ctx.identity_matrix()
        ctx.translate(self.width * .5 + 0.0 * r, self.height * .2)
        draw(self.start_lights >= 2)
        ctx.identity_matrix()
        ctx.translate(self.width * .5 + 3.5 * r, self.height * .2)
        draw(self.start_lights >= 3)

    def close(self):
        self._bvr.close()


def from_pil(im: Image, alpha: float=1.0, format: cairo.Format=cairo.FORMAT_ARGB32) -> cairo.ImageSurface:
    """
    :param im: Pillow Image
    :param alpha: 0..1 alpha to add to non-alpha images
    :param format: Pixel format for output surface
    """
    assert format in (
        cairo.FORMAT_RGB24,
        cairo.FORMAT_ARGB32,
    ), f"Unsupported pixel format: {format}"
    #if 'A' not in im.getbands():
    #    im.putalpha(int(alpha * 256.))
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    arr = bytearray(im.tobytes('raw', 'BGRA'))
    surface = cairo.ImageSurface.create_for_data(arr, format, im.width, im.height)
    return surface
