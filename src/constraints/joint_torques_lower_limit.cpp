#include "robotoc/constraints/joint_torques_lower_limit.hpp"


namespace robotoc {

JointTorquesLowerLimit::JointTorquesLowerLimit(const Robot& robot)
  : ConstraintComponentBase(),
    dimc_(robot.jointEffortLimit().size()),
    umin_(-robot.jointEffortLimit()) {
}


JointTorquesLowerLimit::JointTorquesLowerLimit()
  : ConstraintComponentBase(),
    dimc_(0),
    umin_() {
}


JointTorquesLowerLimit::~JointTorquesLowerLimit() {
}


KinematicsLevel JointTorquesLowerLimit::kinematicsLevel() const {
  return KinematicsLevel::AccelerationLevel;
}


bool JointTorquesLowerLimit::isFeasible(Robot& robot, 
                                        const ContactStatus& contact_status, 
                                        ConstraintComponentData& data, 
                                        const SplitSolution& s) const {
  for (int i=0; i<dimc_; ++i) {
    if (s.u.coeff(i) < umin_.coeff(i)) {
      // std::cerr << "isFeasible: infieasible at i = " << i << std::endl;
      return false;
    }
  }
  return true;
}


void JointTorquesLowerLimit::setSlack(Robot& robot, 
                                      const ContactStatus& contact_status, 
                                      ConstraintComponentData& data, 
                                      const SplitSolution& s) const {
  data.slack = s.u - umin_;
  // std::cerr << "setSlack: slack = " << data.slack(9) << std::endl;
}


void JointTorquesLowerLimit::evalConstraint(Robot& robot, 
                                            const ContactStatus& contact_status, 
                                            ConstraintComponentData& data, 
                                            const SplitSolution& s) const {
  data.residual = umin_ - s.u + data.slack;
  computeComplementarySlackness(data);
  data.log_barrier = logBarrier(data.slack);
  // std::cerr << "evalConstraint: residual = " << data.residual(9) << std::endl;
}


void JointTorquesLowerLimit::evalDerivatives(
    Robot& robot, const ContactStatus& contact_status, 
    ConstraintComponentData& data, const SplitSolution& s, 
    SplitKKTResidual& kkt_residual) const {
  kkt_residual.lu.noalias() -= data.dual;
  // std::cerr << "evalDerivatives: dual = " << data.dual(9) << ", kkt_residual = " << kkt_residual.lu(9) << std::endl;
}


void JointTorquesLowerLimit::condenseSlackAndDual(
    const ContactStatus& contact_status, ConstraintComponentData& data, 
    SplitKKTMatrix& kkt_matrix, SplitKKTResidual& kkt_residual) const {
  kkt_matrix.Quu.diagonal().array()
      += data.dual.array() / data.slack.array();
  computeCondensingCoeffcient(data);
  kkt_residual.lu.noalias() -= data.cond;
  // std::cerr << "condenseSlackAndDual: dual / slack = " << data.dual(9) / data.slack(9)
  //           << ", Quu = " << kkt_matrix.Quu.diagonal().array()(9) 
  //           << ", cond = " << data.cond(9)
  //           << ", kkt_residual = " << kkt_residual.lu(9)
  //           << std::endl;
}


void JointTorquesLowerLimit::expandSlackAndDual(
    const ContactStatus& contact_status, ConstraintComponentData& data, 
    const SplitDirection& d) const {
  data.dslack = d.du - data.residual;
  computeDualDirection(data);
  // std::cerr << "expandSlackAndDual: dslack = " << data.dslack(9) 
  //           << ", du = " << d.du(9) 
  //           << ", residual = " << data.residual(9)
  //           << std::endl;
}


int JointTorquesLowerLimit::dimc() const {
  return dimc_;
}

} // namespace robotoc