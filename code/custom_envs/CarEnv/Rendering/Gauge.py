from cairo import Context
import numpy as np
from .Rendering import stroke_fill


class Gauge:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.radius = 50.

    def draw(self, ctx: Context, val):
        ctx.arc(0.0, 0.0, self.radius, 0.0, 2 * np.pi)
        stroke_fill(ctx, (0., 0., 0.), (1., 1., 1.))

        start_angle = 2 * np.pi - 1
        stop_angle = 1

        majors = np.linspace(start_angle, stop_angle, 11)

        ctx.set_font_size(9.5)

        for i, angle in enumerate(majors):
            v = np.array([np.sin(angle), np.cos(angle)])
            ctx.move_to(*(v * self.radius))
            ctx.line_to(*(v * .8 * self.radius))
            stroke_fill(ctx, (0., 0., 0.), None)
            if i % 2 == 0:
                ctx.move_to(*(v * .63 * self.radius))
                text = str(int((self.max_val - self.min_val) * (i / (len(majors) - 1))))
                x_offset = ctx.text_extents(text).width / 2
                y_offset = ctx.text_extents(text).height / 2
                ctx.rel_move_to(-x_offset, y_offset)
                ctx.show_text(text)

        ctx.move_to(-11, 35)
        ctx.show_text("km/h")

        # Needle
        val = np.clip(val, self.min_val, self.max_val)
        rel_val = (val - self.min_val) / (self.max_val - self.min_val)
        needle_angle = (stop_angle - start_angle) * rel_val + start_angle
        v = np.array([np.sin(needle_angle), np.cos(needle_angle)])
        ctx.move_to(*(v * self.radius * .8))
        ctx.line_to(*(v * self.radius * -.12))
        stroke_fill(ctx, (1., 0., 0.), None)

        # Center point needle
        ctx.arc(0.0, 0.0, self.radius * .06, 0., 2 * np.pi)
        stroke_fill(ctx, None, (1.0, 0.0, 0.0))
