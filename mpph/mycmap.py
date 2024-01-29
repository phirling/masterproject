from matplotlib.colors import LinearSegmentedColormap

dusk_colors = ["firebrick", "orangered","peachpuff","skyblue" ,"steelblue"] # wheat
dusk_colors_dark = ["firebrick", "orangered","lightsalmon","skyblue" ,"steelblue"] # wheat

cmap_dusk = LinearSegmentedColormap.from_list("dusk", dusk_colors)