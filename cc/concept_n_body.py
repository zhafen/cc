import rebound

import augment

########################################################################

class Simulation( object ):

    @augment.store_parameters
    def __init__(
        self,
        concept_map,
        r_c = 5.,
        rep_power = 3.,
        att_power = 1.,
        inital_dims = ( 10., 10., 10. ),
        inital_vdims = ( 2., 2., 2. )
    ):
        '''The force between two particles is the derivative of
        V(r) = M * a * r ** -rep_power - M * r ** -att_power
        Where M is the mass of the other particle.
        When M >> m, a = (att_power/rep_power) * r_c**(rep_power - att_power)
        where r_c is the circular orbit.
        '''

        # Initialize the simulation
        self.sim = rebound.Simulation()

        # Setup particles
        for c in concept_map.concepts:
            
            x, y, z = [
                np.random.uniform( -length / 2., length / 2. )
                for length in initial_dims
            ]
            vx, vy, vz = [
                np.random.uniform( -vlength / 2., vlength / 2. )
                for vlength in initial_vdims
            ]

            self.sim.add(
                m = concept_map.weights[c],
                x = x, y = y, z = z,
                vx = vx, vy = vy, vz = vz,
            )

        # Move to center-of-momentum frame
        sim.move_to_com()

        # Setup repulsive force
        def scaled_repulsive_force( r ):

            prefactor = att_power * r_c**(rep_power - att_power)

            force = prefactor * r**( -rep_power - 1 )

            return force

        # Add additional forces
        def repulsive_force( sim ):

            ps = sim.contents.particles

            # Loop through particles
            for i, p in enumerate( ps ):
                net_force = 0.
                # Loop through other particles
                for j, p_e in enumerate( ps ):

                    assert False, "Need to calc r."

                    net_force += p_e.m * scaled_repulsive_force( r )

    ########################################################################

