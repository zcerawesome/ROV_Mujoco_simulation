// C ABI around the X19-Core Controller so Python (ctypes) can drive the exact
// same C++ that runs on the ROV. Keep this file free of control logic.
#include "controller.hpp"

extern "C" {

Controller *ctrl_new(double k_p, double k_i, double k_d) {
  return new Controller((Weights){.k_p = k_p, .k_i = k_i, .k_d = k_d});
}

void ctrl_free(Controller *c) { delete c; }

void ctrl_set_target(Controller *c, double target) { c->target = target; }

double ctrl_get_target(Controller *c) { return c->target; }

void ctrl_set_weights(Controller *c, double k_p, double k_i, double k_d) {
  c->weights = (Weights){.k_p = k_p, .k_i = k_i, .k_d = k_d};
}

void ctrl_record_measurement(Controller *c, double measurement) {
  c->record_measurement(measurement);
}

double ctrl_get_measurement(Controller *c) { return c->get_measurement(); }

double ctrl_compute(Controller *c, double measurement, double dt) {
  return c->compute(measurement, dt);
}

double ctrl_compute_filtered(Controller *c, double dt) {
  return c->compute_filtered(dt);
}

double ctrl_get_integral(Controller *c) { return c->integral; }

void ctrl_reset_integral(Controller *c) { c->integral = 0; }

}
