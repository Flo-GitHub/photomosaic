#todavía solo el ejemplo de la documentación...

from skimage import img_as_float
import photomosaic as pm


# (1) Prepare the Image

# Size the image to be evenly divisible by the tiles.
image = img_as_float(image)

# Use perceptusally uniform colorspace for all analysis.
converted_img = pm.perceptual(image)

# (2)Optional: Optimize the Color Palette
# Adapt the color palette of the image to resemble the palette of the pool.
adapted_img = pm.adapt_to_pool(converted_img, pool)

# (3) Partition Tiles
scaled_img = pm.rescale_commensurate(adapted_img, grid_dims=(30, 30), depth=1)
tiles = pm.partition(scaled_img, grid_dims=(30, 30), depth=1)

#Optionally, visualize the tile layout. 
# (Recall that scaled_img is represented in the perceptually-uniform color space;
#  we have to convert it back to RGB for visualization.)
annotated_img = pm.draw_tile_layout(pm.rgb(scaled_img), tiles)
# Save the result with imsave, or plot with matplotlib:
import matplotlib.pyplot as plt
plt.imshow(annotated_img)

# (4) Match Tiles to Pool Images
import numpy

# Reshape the 3D array (height, width, color_channels) into
# a 2D array (num_pixels, color_channels) and average over the pixels.
tile_colors = [numpy.mean(scaled_img[tile].reshape(-1, 3), 0)
               for tile in tiles]

# Match a pool image to each tile.
match = pm.simple_matcher(pool) # NOTE: A different strategy for characterizing the tiles and the pool could be used
matches = [match(tc) for tc in tile_colors]

# (5) Draw Mosaic
import numpy
canvas = numpy.ones_like(scaled_img)  # white canvas

# Draw the mosaic.
mos = pm.draw_mosaic(canvas, tiles, matches)

cache = {}
mos1 = pm.draw_mosaic(canvas1, tiles1, matches1, resized_copy_cache=cache)
# Now cache is filled with resized copies of any images used in ``mos1``.

# This will be faster:
mos2 = pm.draw_mosaic(canvas2, tiles2, matches2, resized_copy_cache=cache)