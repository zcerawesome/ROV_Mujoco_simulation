// Vendored from purduerov/X19-Core, branch depth_control (db2be30):
//   nodes/cpp/depth_control/controller.cpp
// Changes from upstream are marked "SIM FIX" (see upstream_fixes.patch).
#include "controller.hpp"

double Controller::compute(double measurement, double dt) {
  double error = target - measurement;

  // SIM FIX: upstream had `double p = weights.k_p;`, i.e. a constant output
  // that ignores the error, so P control never responded to depth.
  double p = weights.k_p * error;

  integral += error * dt;
  double i = weights.k_i * integral;

  // SIM FIX: last_error starts at 0, so the first derivative was
  // (error - 0) / dt -- a full-scale "derivative kick" on enable.
  if (num_items == 0) {
    last_error = error;
  }
  double derivative = (error - last_error) / dt;
  double d = weights.k_d * derivative;

  double output = p + i + d;

  if (output > MAX_OUTPUT) {
    output = MAX_OUTPUT;
    integral -= error * dt;
  } else if (output < MIN_OUTPUT) {
    output = MIN_OUTPUT;
    integral -= error * dt;
  }

  last_error = error;
  num_items++;

  return output;
}

double Controller::compute_filtered(double dt) {
  return this->compute(moving_avg, dt);
}

void Controller::record_measurement(double measurement) {
  // SIM FIX: moving_avg starts at 0, so without seeding the filter the
  // first few outputs think the ROV is at the surface and dive hard.
  if (!has_measurement) {
    moving_avg = measurement;
    has_measurement = true;
    return;
  }
  // s_t     = \alpha    * x_t         + (1 - \alpha)    * s_{t-1}
  moving_avg = EMA_Alpha * measurement + (1 - EMA_Alpha) * moving_avg;
}

double Controller::get_measurement() {
  return moving_avg;
}
