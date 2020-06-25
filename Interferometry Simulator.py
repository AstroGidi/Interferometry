from pylab import *
import matplotlib.pyplot as plt
plt.style.use('dark_background')
colors = ["maroon", "peru", "olivedrab", "darkseagreen", "lightseagreen", "lightskyblue", "slateblue", "mediumorchid"] # Pallette
fontsize = 16

#### Units and conversions ####
day2sec = 24 * 60 * 60
hr2sec = 3600
min2sec = 60
rho_h2O_cgs = 1 # Liquid Water density [g/cm^3]
g_bar_cgs = 981 # Average grvitation on the surface of the earth [cgs]
c = 3e8 # Speed of light [SI]
h = 6.626e-34 # Planck's constant [SI]
kB = 1.38e-23 # Boltzmann's constant [SI]
mas2deg = 2.7777777777778e-7 # degree to miliarcseconds
as2deg = mas2deg * 1000 # degrees to arcseconds
meter2pc = 3.240779289e-17 # meter to parsec
meter2au = 6.68459e-12 # meter to AU

#### Useful functions ####

# Modulus of a complex number calculator #
def complex_modulus(z):

    a = z.real
    b = z.imag

    return np.sqrt(a ** 2 + b ** 2)

# Plane wave #
def psi(psi_0, nu, t, t_phase):

    # amp: amplitude [arbitrary units]
    # nu: frequency [1 / sec]
    # t: propagation time [sec]
    # t_phase: temporal phase [sec]

    return psi_0 * exp(1j * 2 * pi * nu * (t + t_phase))

# Gaussian generator #
def gaussian(x, b, a, mu, sigma):

    # x: 1D spatial coordinate grid
    # b: DC term [units of amplitude]
    # a: amplitude [arbitrary units]
    # mu: Gaussian centroid [units of x]
    # sigma: Gaussian width [units of x]

    return b + a * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

# U-V plane sampling rotation matrix Eq. 2.4 from http://www.jmmc.fr/mirrors/obsvlti/book/Segransan_1.pdf #
def u_v_rotMatrix(h, declination, wl):

    # h: Hour angle [deg]
    # declination: source declination [deg]
    # wl: wavelength [m]

    return (1 / wl) * np.array([[sin(deg2rad(h)), cos(deg2rad(h)), 0.], [(- sin(deg2rad(declination)) * cos(deg2rad(h))), (sin(deg2rad(declination)) * sin(deg2rad(h))), (cos(deg2rad(declination)))],\
                               [(cos(deg2rad(declination)) * cos(deg2rad(h))), (- cos(deg2rad(declination)) * sin(deg2rad(h))), (sin(deg2rad(declination)))]])

# Generate 2D gaussian disk #
def gaussian_disk(sigma_grid, a, b, sigma, x0, y0):

    # sigma_grid: 1D coordinate will be turned into 2D mesh grid [deg^2]
    # a: amplitude [arbitrary units]
    # b: DC term [units of a]
    # sigma: Gaussian disk width [units of sigma_grid]
    # x0, y0: Centroid position [units of sigma_grid]

    x, y = np.meshgrid(sigma_grid, sigma_grid)

    I_sigma_2D = b + a * np.exp(-(((np.sqrt((x - x0) ** 2. + (y - y0) ** 2.)) / (2 * sigma)) ** 2))

    return I_sigma_2D

# Generate 2D Gaussian ring #
def gaussian_ring(sigma_grid, a, b, x0, y0, sigma, radius):

    # sigma_grid: 1D coordinate will be turned into 2D mesh grid [deg^2]
    # a: amplitude [arbitrary units]
    # b: DC term [units of a]
    # x0, y0: Centroid position [units of sigma_grid]
    # sigma: Gaussian disk width [units of sigma_grid]
    # radius: radius of the ring [units of sigma_grid]

    x, y = np.meshgrid(sigma_grid, sigma_grid)

    I_sigma_2D = b + a * np.exp(-((np.sqrt((x - x0) ** 2. + (y - y0) ** 2.) - radius) / (2 * sigma)) ** 2)

    return I_sigma_2D

# Generate 2D uniform disk #
def uniform_disk(sigma_grid, a, radius, x0, y0):

    # sigma_grid: 1D coordinate will be turned into 2D mesh grid [deg^2]
    # a: amplitude [arbitrary units]
    # radius: radius of the ring [units of sigma_grid]
    # x0, y0: Centroid position [units of sigma_grid]

    x, y = np.meshgrid(sigma_grid, sigma_grid)

    I_sigma_2D = np.zeros((len(sigma_grid), len(sigma_grid)))
    I_sigma_2D[np.where(np.sqrt((x - x0) ** 2. + (y - y0) ** 2.) < radius)] = a

    return I_sigma_2D

# Generate 2D uniform ring #
def uniform_ring(sigma_grid, a, outer_radius, width, x0, y0):

    # sigma_grid: 1D coordinate will be turned into 2D mesh grid [deg^2]
    # a: amplitude [arbitrary units]
    # outer_radius: Outer radius of the ring [units of sigma_grid]
    # width: Width of the ring [units of sigma_grid]
    # x0, y0: Centroid position [units of sigma_grid]

    x, y = np.meshgrid(sigma_grid, sigma_grid)

    I_sigma_2D = np.zeros((len(sigma_grid), len(sigma_grid)))

    I_sigma_2D[np.where(np.sqrt((x - x0) ** 2. + (y - y0) ** 2.) < outer_radius)] = a
    I_sigma_2D[np.where(np.sqrt((x - x0) ** 2. + (y - y0) ** 2.) < outer_radius - width)] = 0.

    return I_sigma_2D

# Shape generator #
def shape_gen(shape, sigma_grid, args, x0, y0):

    # Generate desired 2D shape #
    if shape == "gaussian_disk":
        return gaussian_disk(sigma_grid, *args, x0, y0)
    if shape == "gaussian_ring":
        return  gaussian_ring(sigma_grid, *args, x0, y0)
    if shape == "uniform_disk":
        return  uniform_disk(sigma_grid, *args, x0, y0)
    if shape == "uniform_ring":
        return  uniform_ring(sigma_grid, *args, x0, y0)
    else:
        print("Shape doesn't exist. Possible options: gaussian_disk, gaussian_ring, uniform_disk, uniform_ring")
        exit()

# Plot u-v plane sampling due to earth's rotation (for average wavelength) #
def plot_u_v_earthRot(Baselines, u_orientation, dec, Hr_angle):

    # Baselines: VECTOR of Baseline lengths [m]
    # u_orientation: VECTOR of clockwise rotation angles of the corresponding baseline [deg]
    # dec: Source declination
    # Hr_angle: VECTOR of hour angles [deg]

    aperture_circle = plt.Circle((0., 0.), max(Baselines), color='r', fill=False)

    ax = plt.gca()
    ax.cla()

    ax.set_xlim((- 2 * max(Baselines), 2 * max(Baselines)))
    ax.set_ylim((- 2 * max(Baselines), 2 * max(Baselines)))

    ax.add_artist(aperture_circle)
    plt.title("Earth-rotation over h = {0}-{1}$^o$, $\delta$ = {2}$^o$".format(round(Hr_angle[0], 2), round(Hr_angle[-1], 2), round(dec, 2)))
    plt.xlabel("u [m]", fontsize = fontsize)
    plt.ylabel("v [m]", fontsize = fontsize)
    plt.tick_params(labelsize = fontsize)

    for b in range(len(Baselines)):

        for i in range(len(Hr_angle)):

            u, v, w = Baselines[b] * cos(deg2rad(u_orientation[b])), Baselines[b] * sin(deg2rad(u_orientation[b])), 0.

            u_v_rotMat = u_v_rotMatrix(Hr_angle[i], dec, 1.)

            u_proj_u, u_proj_v, _ = u_v_rotMat @ np.array([u, v, w])

            if i == 0:
                plt.plot(u_proj_u, u_proj_v, label='B = {0} [m], $\\theta_B$ = {1} [deg]'.format(Baselines[b], round(u_orientation[b], 2)), color=colors[b])
                plt.plot(- u_proj_u, - u_proj_v, '.', color=colors[b])
            else:
                plt.plot(u_proj_u, u_proj_v, '.', color=colors[b])
                plt.plot(- u_proj_u, - u_proj_v, '.', color=colors[b])

    plt.legend(fontsize = 6)
    plt.show()

# Plot u-v plane sampling due to wavelength synthesis (for average hour angle) #
def plot_u_v_wlSynthesis(Baselines, u_orientation, wavelengths, dec, Hr_angle):

    # Baselines: VECTOR of Baseline lengths [m]
    # u_orientation: VECTOR of clockwise rotation angles of the corresponding baseline [deg]
    # wavelengths: Spectral range VECTOR of the interferometer [m]
    # dec: Source declination
    # Hr_angle: SINGLE value of hour angle

    # aperture_circle = plt.Circle((0., 0.), 1e-6 * max(Baselines) / min(wavelengths), color='r', fill=False)

    ax = plt.gca()
    ax.cla()

    ax.set_xlim((- 2e-6 * max(Baselines) / min(wavelengths), 2e-6 * max(Baselines) / min(wavelengths)))
    ax.set_ylim((- 2e-6 * max(Baselines) / min(wavelengths), 2e-6 * max(Baselines) / min(wavelengths)))

    # ax.add_artist(aperture_circle)
    plt.title("Wavelength synthesis over {0}-{1} $\mu m$, $\delta$ = {2}$^o$, h = {3}$^o$".format(round(wavelengths[0] * 1e6, 0), round(wavelengths[-1] * 1e6, 0), round(dec, 2), round(Hr_angle, 2)))
    plt.xlabel("$B_{x}$ [$M\lambda$]", fontsize = fontsize)
    plt.ylabel("$B_{y}$ [$M\lambda$]", fontsize = fontsize)
    plt.tick_params(labelsize = fontsize)

    for b in range(len(Baselines)):

        for w in range(len(wavelengths)):

            u, v, z = Baselines[b] * cos(deg2rad(u_orientation[b])), Baselines[b] * sin(deg2rad(u_orientation[b])), 0.

            u_v_rotMat = u_v_rotMatrix(Hr_angle, dec, wavelengths[w])

            u_proj_u, u_proj_v, _ = u_v_rotMat @ np.array([u, v, z]) * 1e-6

            if w == 0:
                plt.plot(u_proj_u, u_proj_v,
                         label='B = {0} [m], $\\theta_B$ = {1} [deg]'.format(Baselines[b], round(u_orientation[b], 2)),
                         color=colors[b])
                plt.plot(- u_proj_u, - u_proj_v, '.', color=colors[b])
            else:
                plt.plot(u_proj_u, u_proj_v, '.', color=colors[b])
                plt.plot(- u_proj_u, - u_proj_v, '.', color=colors[b])

    plt.legend(fontsize=6)
    plt.show()

def run(wavelengths, beta, detector_coords, Baselines, source_params, plotting_knobs):

    # wavelengths: Spectral range VECTOR of the interferometer [m]
    # detector_coords: 1D coords VECTOR of the detector surface [m]
    # beta: angle of incidence of the beam on the detector [deg]
    # Baselines: list of tuples of shape: (Baseline length [m], Baseline clockwise rotation with respect to u-v plane [deg])
    # beta: Angle of incidence of the beam onto the detector [deg]
    # source_params: DICTIONARY of shape: ["dec"] = source declination [deg], ["lat"] = observatory latitude [deg], ["h"] = VECTOR of Hour Angles of the source, ["dist"] = source distance [pc], ["characteristic_scale"] = characteristic scale of the object's geometry (sigma for Gaussian shapes, outer radius for uniform shapes) [as] \
    # ["shape"]: STRING of desired shape of object (uniform_ring, uniform_disk, gaussian_ring, gaussian_disk), ["shape_params"]: LIST of parameters of the shape, as required by each function except for centroid position and sigma_grid
    # plotting_knobs: DICTIONARY of booleans for different plotting options: ["shape"] = plots generated 2D shape, ["uv_rot"] = plots the sampled u-v plane due to the rotation of the earth (avg. wavelength), ["uv_wlSynth"] = plots the sampled u-v plane due wavelength synthesis (avg. Hr angle)
    # ["u_vec_mag"]: plot the magnitude of he u,v vectors as a function of wavelength for each baseline, ["visibilities"]: plot visibilities as functions of u * characteristic_scale and wavelength for all baselines, ["phases"]: plot phases as functions of u * characteristic_scale and wavelength for all baselines

    #### Defining required quantities (nomenclature according to Busch book) ####
    lat, dec, Hr_angles = source_params["lat"], source_params["dec"], source_params["Hr_angles"] # observatory lat [deg], target dec [deg], target Hour angles [deg]
    characteristic_scale, dist = source_params["characteristic_scale"], source_params["dist"] # Characteristic scale of the shape [deg], distance of the shape [pc]
    sigma_grid = np.linspace(- 2 * characteristic_scale, 2  * characteristic_scale, 500) # Defining grid of coordinates to be observed [deg]
    sigma_u, sigma_v = np.meshgrid(sigma_grid, sigma_grid) # Generating a meshgrid of the u,v plane
    u_0, v_0 = source_params["source_center_lm"][0], source_params["source_center_lm"][1] # Source center in the l,m plane [deg]
    theta_0 = np.arctan2(u_0, v_0) # Angular radial distance of the source center from the phase center (l, m = 0) (1.20)
    s = np.array(2 * sin(deg2rad(beta)) / wavelengths) # Spatial frequency of the fringes as function of frequency

    # Generate desired 2D shape #
    I_sigma_2D = shape_gen(source_params["shape"], sigma_grid, source_params["shape_params"], u_0, v_0)
    F0 = sum(sum(I_sigma_2D))
    object_visibility = I_sigma_2D / sum(sum(I_sigma_2D))  # Eq 1.49

    # Generating a list of spatial frequencies U (Eq. 1.22)
    B, B_rotation = np.array([Baselines[i][0] for i in range(len(Baselines))]), np.array([Baselines[i][1] for i in range(len(Baselines))])

    u_2D_vec = np.array([[(B[b] / wavelengths[w]) * np.array([cos(deg2rad(B_rotation[b])), sin(deg2rad(B_rotation[b]))]) for w in range(len(wavelengths))] for b in range(len(B))]) # Matrix of u vectors for all baselines and wavelengths (Eq. 1.22)
    u_sigma_2D = np.array([[exp(-1j * 2 * pi * ((u_2D_vec[b][w][0]) * sigma_u + u_2D_vec[b][w][1] * sigma_v)) for w in range(len(wavelengths))] for b in range(len(B))]) # Generating spatial wavelength coverage of of each given Baseline at every wavelength across the u-v plane

    u_2D_vec_avg = np.array([(B[b] / average(wavelengths)) * np.array([cos(deg2rad(B_rotation[b])), sin(deg2rad(B_rotation[b]))]) for b in range(len(B))]) # 1D vector of baselines at the average sampled wavelength
    u_sigma_2D_avg = np.array([real(exp(-1j * 2 * pi * ((u_2D_vec_avg[b][0]) * sigma_u + u_2D_vec_avg[b][1] * sigma_v))) for b in range(len(B))]) # Generating spatial wavelength coverage of of each given Baseline at the average wavelength value

    #### Generating visibilities ####
    Vu_2D = np.array([[sum(sum(object_visibility * u_sigma_2D[b][w])) for w in range(len(wavelengths))] for b in range(len(Baselines))]) # Correlated flux for all Baselines and wavelengths
    i_x_2D = np.array([[F0 + real(real(Vu_2D[b][w]) * exp(1j * 2 * pi * s[w] * detector_coords + imag(Vu_2D[b][w]))) for w in range(len(wavelengths))] for b in range(len(Baselines))]) # Fringe pattern for every Baseline and wavelength
    visibilities_2D = np.array([[real(Vu_2D[b][w]) for w in range(len(wavelengths))] for b in range(len(Baselines))]) # Visibility for each Baseline and wavelength
    phases_2D = np.array([[imag(Vu_2D[b][w]) for w in range(len(wavelengths))] for b in range(len(Baselines))]) # Phase for each Baseline and wavelength

    ######## PLOTTING ######## ######## PLOTTING ######## ######## PLOTTING ######## ######## PLOTTING ######## ######## PLOTTING ######## ######## PLOTTING ######## ######## PLOTTING ######## ######## PLOTTING ########

    if plotting_knobs["shape"] == True:

        fig, ax = plt.subplots(1)
        ax.set_title(source_params["shape"], fontsize=fontsize)
        fig.colorbar(ax.imshow(I_sigma_2D, extent = (sigma_grid[0] / mas2deg, sigma_grid[-1] / mas2deg, sigma_grid[0] / mas2deg, sigma_grid[-1] / mas2deg), cmap ='magma', origin='lower', interpolation='nearest'))
        ax.set_xlabel("$\delta$RA [mas]", fontsize=fontsize)
        ax.set_ylabel("$\delta$DEC [mas]", fontsize=fontsize)
        ax.set_xlim(left =max(sigma_grid) / mas2deg, right=min(sigma_grid) / mas2deg)
        ax.set_ylim(bottom=min(sigma_grid) / mas2deg, top =max(sigma_grid) / mas2deg)
        ax.tick_params(labelsize=fontsize)
        fig.show()

    if plotting_knobs["uv_rot"] == True:
        plot_u_v_earthRot(B, B_rotation, dec, Hr_angles)

    if plotting_knobs["uv_wlSynth"] == True:
        plot_u_v_wlSynthesis(B, B_rotation, wavelengths, dec, average(Hr_angles))

    if plotting_knobs["u_vec_mag"] == True:

        for b in range(len(u_2D_vec)):

            u_this_b = np.array([u_2D_vec[b][i][0] for i in range(len(u_2D_vec[b]))])
            v_this_b = np.array([u_2D_vec[b][i][1] for i in range(len(u_2D_vec[b]))])

            plt.subplot(len(u_2D_vec), 1, b + 1)
            plt.plot(wavelengths * 1e6, u_this_b * characteristic_scale, color = 'blue', label = 'u$\Delta\sigma$ for B = {0} [m], $\\theta$ = {1}$^o$'.format(Baselines[b][0], round(Baselines[b][1], 1)))
            plt.plot(wavelengths * 1e6, v_this_b * characteristic_scale, '--', color = 'orange', label = 'v$\Delta\sigma$'.format(Baselines[b][0], round(Baselines[b][1], 1)))
            plt.ylabel("$\\vec{u}\Delta\sigma$", fontsize = fontsize - 2)
            plt.tick_params(labelsize = fontsize - 2)
            plt.legend()

            if b == len(u_2D_vec) - 1:
                plt.xlabel("Wavelength [$\mu$m]", fontsize=fontsize - 2)
                plt.show()

    if plotting_knobs["visibilities"] == True:

        g = 0

        for b in range(len(visibilities_2D)):

            u_this_b = np.array([u_2D_vec[b][i][0] for i in range(len(u_2D_vec[b]))])
            v_this_b = np.array([u_2D_vec[b][i][1] for i in range(len(u_2D_vec[b]))])

            phase_this_b = np.array([visibilities_2D[b][i] for i in range(len(visibilities_2D[b]))])
            g = g + 1

            plt.subplot(len(u_2D_vec), 2, g)
            plt.plot(wavelengths / 1e-6, phase_this_b, color ='goldenrod')
            plt.ylabel("Visibility", fontsize = fontsize - 2)
            plt.tick_params(labelsize = fontsize - 2)

            if b == len(u_2D_vec) - 1:
                plt.xlabel("Wavelength [$\mu$m]", fontsize=fontsize - 2)

            g = g + 1

            plt.subplot(len(u_2D_vec), 2, g)
            plt.plot(sqrt(u_this_b ** 2 + v_this_b ** 2) * characteristic_scale, phase_this_b, color ='skyblue')
            # plt.plot(u_this_b * characteristic_scale, vis_this_b, color = 'skyblue', label = 'u')
            # plt.plot(v_this_b * characteristic_scale, vis_this_b, color = 'mediumorchid', label = 'v')
            plt.tick_params(labelsize = fontsize - 2)

            # if b == 0:
            #     plt.legend()

            if b == len(u_2D_vec) - 1:
                plt.xlabel("$u\Delta\\sigma$", fontsize=fontsize - 2)
        plt.show()

    if plotting_knobs["phases"] == True:

        g = 0

        for b in range(len(phases_2D)):

            u_this_b = np.array([u_2D_vec[b][i][0] for i in range(len(u_2D_vec[b]))])
            v_this_b = np.array([u_2D_vec[b][i][1] for i in range(len(u_2D_vec[b]))])

            phase_this_b = np.array([rad2deg(phases_2D[b][i]) for i in range(len(phases_2D[b]))])
            g = g + 1

            plt.subplot(len(u_2D_vec), 2, g)
            plt.plot(wavelengths / 1e-6, phase_this_b, color ='olivedrab')
            plt.ylabel("Phase [$^o$]", fontsize = fontsize - 2)
            plt.tick_params(labelsize = fontsize - 2)

            if b == len(u_2D_vec) - 1:
                plt.xlabel("Wavelength [$\mu$m]", fontsize=fontsize - 2)

            g = g + 1

            plt.subplot(len(u_2D_vec), 2, g)
            plt.plot(sqrt(u_this_b ** 2 + v_this_b ** 2) * characteristic_scale, phase_this_b, color ='slateblue')
            # plt.plot(u_this_b * characteristic_scale, vis_this_b, color = 'skyblue', label = 'u')
            # plt.plot(v_this_b * characteristic_scale, vis_this_b, color = 'mediumorchid', label = 'v')
            plt.tick_params(labelsize = fontsize - 2)

            # if b == 0:
            #     plt.legend()

            if b == len(u_2D_vec) - 1:
                plt.xlabel("$u\Delta\\sigma$", fontsize=fontsize - 2)
        plt.show()

######## EXAMPLE ######## ######## EXAMPLE ######## ######## EXAMPLE ######## ######## EXAMPLE ########

source_center_lm = [0. * mas2deg, 0. * mas2deg] # [deg]
Baselines = [(100, 45.), (23, 12.), (33, 82.)] # M, deg
characteristic_scale, dist = 1 * mas2deg, 400. # deg, pc
shape = "gaussian_disk"
shape_params = [1., 0., characteristic_scale / 2.] # look at shape generation functions - the input here has to be all the parameters EXCEPT sigma_grid, x0 and y0
wavelengths = np.linspace(5, 130, 1000) * 1e-6 # [m]
Hr_angle, dec, lat = 28.2576335631872, -21.95482, -24.6273 # [deg]
Hr_angle_vec = np.linspace(Hr_angle - 50., Hr_angle + 16., 50) # [deg]
beta = 50. * as2deg # Beam angle of incidence on the detector [deg]
detector_coords = np.linspace(-0.008, 0.008, 1000) # [m]
source_params = {"source_center_lm": source_center_lm, "characteristic_scale": characteristic_scale, "dist": dist, "shape": shape, "shape_params": shape_params, "lat": lat, "dec": dec, "Hr_angles": Hr_angle_vec}
plotting_knobs = {"shape": True, "uv_rot": True, "uv_wlSynth": True, "u_vec_mag": True, "visibilities": True, "phases": True}

#### PLOTTING KNOBS ####
run(wavelengths, beta, detector_coords, Baselines, source_params, plotting_knobs)
















