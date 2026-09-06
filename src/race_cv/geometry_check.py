"""Cross-check the ROI crop against the geometry that decides who finished.

Cropping the frame is a real win -- see ``RoiConfig`` -- but it is also the
single most dangerous setting in this file, because it fails *invisibly*. A
racer outside the crop is not a low-confidence detection or a dropped frame.
They are not in the input at all. Nothing downstream can report them missing,
because nothing downstream ever knew they existed. That is exactly what the
2025 pipeline did: it cropped away the left ~28% of every frame and no log
line, counter or overlay said so.

The specific way this goes wrong is a crop that disagrees with the course
boundary or the finish line. Those are calibrated separately, in the same
normalized coordinates, and there is nothing forcing them to be consistent --
so a crop tightened to save compute can quietly slice through the region the
boundary says is live course, and the two settings will each look correct on
their own.

So: at startup, sample the boundary and the finish line at many heights and
check every sampled point survives the crop. Warnings are returned rather
than raised -- this must not refuse to start a race -- but they name the exact
setting and the exact value that would fix it.
"""

from __future__ import annotations

from .boundary import _interpolate_x_at_y
from .config import Config
from .detect import normalize_imgsz

SAMPLES = 64


def _roi_rect(config: Config, width: int, height: int):
    """The ROI as (x1, y1, x2, y2) in pixels, or None when disabled.

    Mirrors ``detect.Roi`` exactly, truncation included. Describing a crop
    that differs from the one actually taken -- even by a pixel -- would make
    this check a second source of truth about the same geometry, which is the
    problem it exists to solve.
    """
    roi = config.roi
    if not roi.enabled or not roi.polygon:
        return None
    xs = [p[0] * width for p in roi.polygon]
    ys = [p[1] * height for p in roi.polygon]
    return (max(0, int(min(xs))), max(0, int(min(ys))),
            min(width, int(max(xs))), min(height, int(max(ys))))


def _sample_line(p1, p2, y_lo: float, y_hi: float):
    """Yield (y, x) along a boundary line, only where that line is defined.

    Boundary lines are two calibrated points, not infinite rays. Sampling
    outside their own y-range extrapolates them into nonsense -- the right
    line here runs off to x=1.5 and the left to x=-0.8 -- and would report a
    crop as clipping course that does not exist at that height.
    """
    lo = max(y_lo, min(p1[1], p2[1]))
    hi = min(y_hi, max(p1[1], p2[1]))
    if hi < lo:
        return
    for i in range(SAMPLES + 1):
        y = lo + (hi - lo) * i / SAMPLES
        yield y, _interpolate_x_at_y(p1, p2, y)


def check_roi_covers_course(config: Config, width: int, height: int) -> list[str]:
    """Warn when the ROI crop cuts into live course or the finish line."""
    rect = _roi_rect(config, width, height)
    if rect is None:
        return []
    rx1, ry1, rx2, ry2 = rect
    warnings: list[str] = []

    if config.course_boundary.enabled:
        boundary = config.course_boundary
        left_p1 = (boundary.left_p1[0] * width, boundary.left_p1[1] * height)
        left_p2 = (boundary.left_p2[0] * width, boundary.left_p2[1] * height)
        right_p1 = (boundary.right_p1[0] * width, boundary.right_p1[1] * height)
        right_p2 = (boundary.right_p2[0] * width, boundary.right_p2[1] * height)

        # Only where the boundary is actually defined, and only within the
        # crop's own vertical span -- above the crop there is no course left to
        # protect, which is the whole point of cropping the sky away.
        worst_left = None
        worst_right = None
        for _, x in _sample_line(left_p1, left_p2, ry1, ry2):
            if x < rx1 and (worst_left is None or x < worst_left):
                worst_left = x
        for _, x in _sample_line(right_p1, right_p2, ry1, ry2):
            if x > rx2 and (worst_right is None or x > worst_right):
                worst_right = x

        if worst_left is not None:
            warnings.append(
                f"roi crops at x={rx1 / width:.3f} but course_boundary extends to "
                f"x={worst_left / width:.3f}. The strip between them is live course "
                f"the detector will never see. Set roi.polygon's left edge to "
                f"<= {max(0.0, worst_left / width - 0.02):.3f}, or move the boundary."
            )

        # The top edge. With a rectangular export, cropping the top is a real
        # compute saving (height is no longer padded up to match width), which
        # makes it tempting to push -- and a racer's box top sits well above
        # their feet, so the course's topmost declared point is where their
        # *feet* can be, not where the detector needs to see them. Demand a
        # margin above it, not just clearance.
        course_top = min(p[1] for p in (left_p1, left_p2, right_p1, right_p2))
        margin = 0.10 * height
        if ry1 > course_top - margin:
            warnings.append(
                f"roi crops at y={ry1 / height:.3f} but course_boundary is declared "
                f"from y={course_top / height:.3f}. A racer at the top of the course "
                f"has their head and bib above their feet; keep the crop's top edge "
                f"<= {max(0.0, (course_top - margin) / height):.3f}."
            )
        if worst_right is not None:
            warnings.append(
                f"roi crops at x={rx2 / width:.3f} but course_boundary extends to "
                f"x={worst_right / width:.3f}; that strip is invisible to the "
                f"detector."
            )

    # The finish line itself: a crossing is only detected for a racer the
    # detector saw, so the line must lie inside the cropped region across the
    # full width the racers use.
    line = config.finish_line
    for label, point in (("p1", line.p1), ("p2", line.p2)):
        px, py = point[0] * width, point[1] * height
        # The line is deliberately anchored off-frame vertically; only the
        # horizontal span and the on-screen part of the line matter here.
        if 0 <= py <= height and not (rx1 <= px <= rx2):
            warnings.append(
                f"finish_line.{label} sits at x={point[0]:.3f}, outside the roi crop "
                f"({rx1 / width:.3f}-{rx2 / width:.3f}). A racer crossing there "
                f"cannot be detected."
            )
    return warnings


def describe_roi(config: Config, width: int, height: int) -> str | None:
    """One line saying what the crop keeps, and what it buys.

    Cropping does not make a fixed-input CoreML model faster -- the region is
    letterboxed up to the model's one input size either way. What it buys is
    resolution: the same runner lands on more pixels.
    """
    rect = _roi_rect(config, width, height)
    if rect is None:
        return None
    rx1, ry1, rx2, ry2 = rect
    crop_w, crop_h = rx2 - rx1, ry2 - ry1
    in_w, in_h = normalize_imgsz(config.model.imgsz)
    before = min(in_w / width, in_h / height)
    after = min(in_w / crop_w, in_h / crop_h)
    # Two numbers, because the relative one alone misleads with a rectangular
    # export: "x1.38 what they did full-frame" compares against this same
    # model uncropped, which nobody runs. The absolute scale is what decides
    # whether a bib is legible, and whether it matches what the weights were
    # trained on (x0.667 for a 1920-wide frame into a 1280 square).
    return (
        f"ROI crop {int(crop_w)}x{int(crop_h)} of {width}x{height} "
        f"(x {rx1 / width:.3f}-{rx2 / width:.3f}, y {ry1 / height:.3f}-{ry2 / height:.3f}) "
        f"into a {in_w}x{in_h} model input; objects at x{after:.3f} of native "
        f"(x{after / before:.2f} what this model would see uncropped)"
    )
