def broadcast_time(time):
    return time.reshape(-1,1,1,1)


def interpolate(target, noise, time):
    time = broadcast_time(time)
    return (1-time)*noise + time*target


def velocity_target(target, noise):
    return target-noise


def clean_from_velocity(state, velocity, time):
    return state+(1-broadcast_time(time))*velocity
