
class Collision():
    def __init__(self, sim, p_id):
        self.sim = sim
        self._p = p_id

    def collision_checks(self, collision, closest_check, body_A, body_B, link_A=-1, link_B=-1):
        collision += self._p.getContactPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                              physicsClientId=self.sim.client_id)
        closest_check += self._p.getClosestPoints(bodyA=body_A, bodyB=body_B, linkIndexA=link_A, linkIndexB=link_B,
                                                  distance=0.001, physicsClientId=self.sim.client_id)
        return collision, closest_check

    def object_contact(self, obj_id):
        collision = []
        closest_check = []
        collision += self._p.getContactPoints(bodyA=self.sim.robot_id, bodyB=obj_id,
                                              physicsClientId=self.sim.client_id)
        closest_check += self._p.getClosestPoints(bodyA=self.sim.robot_id, bodyB=obj_id, distance=0.001,
                                                  physicsClientId=self.sim.client_id)
        return True if collision or closest_check else False

    def is_shelf_collision(self):
        collision = []
        closest_check = []
        collision += self._p.getContactPoints(bodyA=self.sim.robot_id, bodyB=self.sim.shelf_id,
                                              physicsClientId=self.sim.client_id)
        closest_check += self._p.getClosestPoints(bodyA=self.sim.robot_id, bodyB=self.sim.shelf_id, distance=0.001,
                                                  physicsClientId=self.sim.client_id)
        return True if collision or closest_check else False

