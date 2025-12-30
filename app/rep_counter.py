from .config import (
    DIP_BOTTOM_ANGLE,
    DIP_TOP_ANGLE,
    MIN_TIME_BETWEEN_REPS,
    PULLUP_BOTTOM_THRESH,
    PULLUP_TOP_THRESH,
    SMOOTHING_ALPHA,
)


def exponential_smooth(new_value, prev_value, alpha):
    if prev_value is None:
        return new_value
    return alpha * new_value + (1.0 - alpha) * prev_value


class RepCounterPullup:
    """
    Simple FSM for pull-up reps based on LEFT shoulder vertical position.
    """

    def __init__(self):
        self.state = "WAITING_FOR_REP"
        self.rep_count = 0
        self.last_rep_time = 0.0
        self.smooth_y = None

    def update(self, y_norm, timestamp):
        if y_norm is None:
            return self.rep_count, self.state, False

        self.smooth_y = exponential_smooth(y_norm, self.smooth_y, SMOOTHING_ALPHA)
        y = self.smooth_y
        rep_incremented = False

        if self.state == "WAITING_FOR_REP":
            if y > PULLUP_BOTTOM_THRESH:
                self.state = "AT_BOTTOM"

        elif self.state == "AT_BOTTOM":
            if y < PULLUP_TOP_THRESH:
                self.state = "AT_TOP"

        elif self.state == "AT_TOP":
            if y > PULLUP_BOTTOM_THRESH:
                if timestamp - self.last_rep_time > MIN_TIME_BETWEEN_REPS:
                    self.rep_count += 1
                    self.last_rep_time = timestamp
                    rep_incremented = True
                self.state = "AT_BOTTOM"

        return self.rep_count, self.state, rep_incremented


class RepCounterDip:
    """
    Simple FSM for dip reps based on LEFT elbow angle.
    """

    def __init__(self):
        self.state = "WAITING_FOR_REP"
        self.rep_count = 0
        self.last_rep_time = 0.0
        self.smooth_angle = None

    def update(self, angle, timestamp):
        if angle is None:
            return self.rep_count, self.state, False

        self.smooth_angle = exponential_smooth(angle, self.smooth_angle, SMOOTHING_ALPHA)
        a = self.smooth_angle
        rep_incremented = False

        if self.state == "WAITING_FOR_REP":
            if a > DIP_TOP_ANGLE:
                self.state = "AT_TOP"

        elif self.state == "AT_TOP":
            if a < DIP_BOTTOM_ANGLE:
                self.state = "AT_BOTTOM"

        elif self.state == "AT_BOTTOM":
            if a > DIP_TOP_ANGLE:
                if timestamp - self.last_rep_time > MIN_TIME_BETWEEN_REPS:
                    self.rep_count += 1
                    self.last_rep_time = timestamp
                    rep_incremented = True
                self.state = "AT_TOP"

        return self.rep_count, self.state, rep_incremented
