from optable import solver
from optable import Ray
from optable import Lens
from optable import OpticalTable

# this solver solves the problem that you have components with
# a specific focal length, but you need to find a 


wl = 780e-9
w0 = 10e-6

r0 = [
    Ray([0, 0, 0], [1, -0.1, 0], wavelength=wl, w0=w0),
    Ray([0, 0, 0], [1, 0, 0], wavelength=wl, w0=w0),
    ]

pos = [10,0,0]
radius = 6
FL = 20

l0 = Lens(pos, radius=radius, focal_length=FL)

rays = r0
components = [l0]
monitors = []

table = OpticalTable()
table.add_components(components)
table.add_monitors(monitors)
table.ray_tracing(rays)

# these are magic, I know these are the components I want
# I can't think of a clever way to find and select them right now.
my_lens = table.components[0] 
my_ray1 = table.rays[0]
my_ray2 = table.rays[1]

# this is the first ray, this is important because the input 
# direction of the ray relative to the lens is important.
print(f"origin{my_ray1.origin}, dir {my_ray1.direction}")

# this ray gets created during / after the collision with the
# lens, so this is the data I'm working with.
print(f"origin{my_ray2.origin}, dir {my_ray2.direction}")

table = OpticalTable()
table.add_components(components)
table.add_monitors(monitors)
table.ray_tracing(rays)

# I am honestly not too sure how to make this work cleanly but since I don't have a good way to
# select things anyway, I will just provide the basics that I know work
# I am transforming things into simple gradients

old_direction = my_ray1.direction[1]/my_ray1.direction[0]
old_bad_direction = my_ray2.direction[1]/my_ray2.direction[0]

print("old bad direction", old_bad_direction)
# -> -0.049750

# let's say we want this to be 0, where do we have to move the lens?

new_direction = 0

my_new_x = solver.solve_move_lens(old_direction,new_direction,my_lens.focal_length)
print("my x position for the lens",my_new_x)

new_y = 10

# create a new lens at the x I calculated and at a new_y for comparison
pos = [my_new_x, new_y, 0]
l1 = Lens(pos, radius=radius, focal_length=FL)

# add another ray at new_y, also two rays that don't get affected for comparison.
r0 = [
    Ray([0, 0, 0], [1, -0.1, 0], wavelength=wl, w0=w0),
    Ray([0, new_y, 0], [1, -0.1, 0], wavelength=wl, w0=w0),
    Ray([0, 0, 0], [1, 0, 0], wavelength=wl, w0=w0),
    Ray([0, new_y, 0], [1, 0, 0], wavelength=wl, w0=w0),
    ]

rays = r0
components = [l0,l1]
monitors = []

table = OpticalTable()
table.add_components(components)
table.add_monitors(monitors)
table.ray_tracing(rays)

if len(table.rays)>1:
    my_new_dir = table.rays[3].direction[1]/table.rays[3].direction[0]
    print("new ray", my_new_dir)
    # -> 0.0004 not exactly zero, something is going on,
    # error gets worse with smaller values on the lens,
    # I guess this module does more advanced things that introduce deviation
    # for my purpose it's good enough.

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    PLOT_TYPE = "Z"
    
    table.render(
        None,
        type=PLOT_TYPE,
        gaussian_beam=True,
        spot_size_scale=1,
        roi=[-15, 30, -10, 20, -10, 10],
    )
    
    plt.savefig("../docs/solve_move.png", dpi=300, bbox_inches="tight")
    plt.show()
