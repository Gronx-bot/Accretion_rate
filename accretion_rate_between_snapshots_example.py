import numpy
from gadget_lib.readsnap_v1_simple import readsnap
import testing
from datetime import datetime
import sys
from astropy.cosmology import Planck15 as cosmo
from astropy.io import fits

def name_conversion(redshift):
	z = round(redshift, 3)
	if z == 0.0: z = "0.000"
	else: z = str(z)

	if len(z) != 5:
		if len(z) != 4 and len(z) != 3:
			raise Exception("Well, it's weird, z is {0}".format(z))
		if len(z) == 4: z += "0"
		if len(z) == 3: z += "00"
		
	return z


def read_haloes(idx, snum, redshift, h, a):
	try:
		z = name_conversion(redshift)
	except Exception as e:
		sys.exit(str(e))


	halodir = 'some_folder'
	filename = halodir+"/"+str(snum)+"/FB15N1024.z"+z+".AHF_halos"
	data = numpy.genfromtxt(filename, skip_header=1)

	#creating arrays 
	p = numpy.zeros([len(idx), 3])
	v = numpy.zeros([len(idx), 3])
	Rvir = numpy.zeros(len(idx))
	Mgas = numpy.zeros(len(idx))
	Mvir = numpy.zeros(len(idx))

	for i in range(len(idx)):
		sel = (data[:,0] == idx[i])

		#everything comoving
		p[i,0] = data[:,5][sel]/h
		p[i,1] = data[:,6][sel]/h
		p[i,2] = data[:,7][sel]/h

		v[i,0] = data[:,8][sel]/a
		v[i,1] = data[:,9][sel]/a
		v[i,2] = data[:,10][sel]/a

		Rvir[i] = data[:,11][sel]/h
		Mgas[i] = data[:,44][sel]/h 
		Mvir[i] = data[:,3][sel]/h

	return p, v, Rvir, Mgas, Mvir


def age_conversion(z_after, z_now):
	#getting time
	t_now = cosmo.age(z_now)
	t_created = cosmo.age(z_after)
	t_lifetyme = t_now - t_created

	#converting from Gyr to Myr
	age = t_lifetyme.value * 1.e+3
	return age


def initial_selection(snap, p_halo, v_halo, Rvir, h, a):
	p_xyz, sel_x, sel_y, sel_z = testing.box(snap['p']/a, p_halo, Rvir) #comoving
	m = testing.selection(snap['m'], sel_x, sel_y, sel_z); m *= 1.e+10; 
	v = numpy.copy(p_xyz); 
	for k in range(3): 
		v[:,k] = testing.selection(snap['v'][:,k]/a, sel_x, sel_y, sel_z) #peculiar velocity -> comoving velocity
		v[:,k] -= v_halo[k]
		p_xyz[:,k] -= p_halo[k] #comoving and centered
	return p_xyz, v, m


def get_flux_spherical(snap, snum, p_halo, V_halo, t_global, n_steps, time_step, cosm):
	
	#getting current cosmic time of snapshot
	h = cosm[0]
	redshift = cosm[1]
	a = cosm[2]
	t_snapshot = cosm[3]

	N_haloes = len(p_halo[:,0])

	#Rb = which_rvir(snum)*a # it's comoving, so I turn it into physical
	Rb = numpy.full(N_haloes, 30) # it's physical

	t = t_global

	#creating arrays
	inflow_tot = numpy.zeros([N_haloes,n_steps])
	outflow_tot = numpy.zeros([N_haloes,n_steps])
	Mgas_tot = numpy.zeros([N_haloes,n_steps])

	for j in range(N_haloes):
		# Selecting particles in halo on global comoving coordinates.
		# Also galaxy doesn't change in size. Rb = fixed physical
		p, v, m = initial_selection(snap, p_halo[j,:], V_halo[j,:], 1.3*Rb[j]/a, h, a)
		
		#comoving -> physical
		for k in range(3):
			p[:,k] = p[:,k]*a
			v[:,k] = v[:,k]*a + h*100*1.e-3*p[:,k]

		for i in range(n_steps):
			dt = (t - t_snapshot)

			#moving selected particles by dt to starting position
			coeff = 1.022e-9 #transform km/s -> kpc/yr
			dt *= 1e+6 #transforming Myr to yr
			p_new = p - v*coeff*dt
			r_new2 = p_new[:,0]*p_new[:,0] + p_new[:,1]*p_new[:,1] + p_new[:,2]*p_new[:,2]

			#choosing particles in the shell at the edge
			r_sel, v_r, m_sel = choose_particles(r_new2, v, p_new, m, Rb[j], time_step)
			
			#calculating flux through the shell
			inflow_tot[j,i], outflow_tot[j,i] = flux_through_shell(r_sel, v_r, m_sel, Rb[j], time_step)

			#calculating gass mass inside the shell
			sel = (r_new2 < Rb[j]**2)
			m_inside = m[sel]
			Mgas_tot[j,i] = numpy.sum(m_inside) 

			t += time_step

		t = t_global
			

	return inflow_tot, outflow_tot, Mgas_tot


def flux_through_shell(r, v_r, m, Rb, time_step):
	#moving particles from starting position to new one 
	coeff = 1.022e-9 #transform km/s -> kpc/yr
	time_step *= 1.e+6 # Myr -> yr
	r_new = r - v_r*coeff*time_step #going backwards in time

	#calculating inflow
	sel = (r < Rb) & (r_new > Rb) 
	m_inflow = m[sel]
	inflow = numpy.sum(m_inflow)/time_step

	#calculating outflow
	sel = (r > Rb) & (r_new < Rb) 
	m_outflow = m[sel]
	outflow = numpy.sum(m_outflow)/time_step

	return inflow, outflow


def choose_particles(r2, v, p, m, Rb, time_step):
	size = time_step*1.e+6 * 300*1.022e-9 # max velocity 300 km/s and dt in Myr
	sel = (r2 >= (Rb - size)**2) & (r2 <= (Rb + size)**2) #inside shell 
	
	#selecting paticles inside this shell
	r_sel = numpy.sqrt(r2[sel])
	m_sel = m[sel]
	v_sel = numpy.zeros([len(m_sel), 3])
	p_sel = numpy.zeros([len(m_sel), 3])
	for k in range(3):
		v_sel[:,k] = v[:,k][sel]
		p_sel[:,k] = p[:,k][sel]

	#calculating the radial velocity of selected particles
	scal = v_sel[:,0]*p_sel[:,0] + v_sel[:,1]*p_sel[:,1] + v_sel[:,2]*p_sel[:,2]
	v_r =  numpy.divide(scal, r_sel)

	return r_sel, v_r, m_sel


def write_to_fits_file(redshift, ids, p, Rvir, Mgas_halo, Mvir, inflow, outflow, 
	Mgas, t_global, n_steps, time_step, filename, folder=''):
	N_haloes = len(Rvir)
	with fits.open(folder+filename, mode='update') as hdul:
		hdul[1].data['redshift'] = numpy.concatenate((hdul[1].data['redshift'], numpy.full(n_steps*N_haloes, redshift)))
		hdul[1].data['id'] = numpy.concatenate((hdul[1].data['id'], numpy.repeat(ids, n_steps)))
		hdul[1].data['x'] = numpy.concatenate((hdul[1].data['x'], numpy.repeat(p[:,0], n_steps)))

		for j in range(n_steps):
			hdul[1].data['inflow'] = numpy.concatenate((hdul[1].data['inflow'], inflow[:,j]))
			hdul[1].data['outflow'] = numpy.concatenate((hdul[1].data['outflow'], outflow[:,j]))
			hdul[1].data['Mgas'] = numpy.concatenate((hdul[1].data['Mgas'], Mgas[:,j]))
			hdul[1].data['time_step'] = numpy.concatenate((hdul[1].data['time_step'], numpy.full(N_haloes, t_global)))
			t_global += time_step

		hdul[1].data['Rvir'] = numpy.concatenate((hdul[1].data['Rvir'], numpy.repeat(Rvir, n_steps)))
		hdul[1].data['Mgas_halo'] = numpy.concatenate((hdul[1].data['Mgas_halo'], numpy.repeat(Mgas_halo, n_steps)))
		hdul[1].data['Mvir'] = numpy.concatenate((hdul[1].data['Mvir'], numpy.repeat(Mvir, n_steps)))
		hdul.flush()


def create_fits_file(filename, folder=''):
	from os import listdir, remove, getcwd
	from os.path import isfile, join

	# removing file if it exists already
	if folder == '':
		path = getcwd()
	elif folder[-1] != '/':
		sys.exit('Incorrect name of the folder while creating fits file')
	else:
		path = folder
	filenames = [file for file in listdir(path) if isfile(join(path, file))]

	if filename in filenames:
		remove(folder+filename)

	# setting names and data types for output
	names = ['redshift', 'id', 'x', 'inflow', 'outflow', 'Rvir',
			 'Mgas_halo', 'Mvir', 'Mgas', 'time_step']
	data_type = {
		'redshift': 'F',
		'id': 'I',
		'x': 'F',
		'inflow': 'F',
		'outflow': 'F',
		'Rvir': 'F',
		'Mgas_halo': 'E',
		'Mvir': 'E',
		'Mgas': 'E',
		'time_step': 'I'
	}

	# creating the file
	col = []
	data = numpy.zeros(0)
	for name in names:
		col += [fits.Column(name=name, format=data_type[name], data=data)]

	hdul = fits.BinTableHDU.from_columns(col)
	hdul.writeto(folder+filename)


def main(SNUM):	
	begintime = datetime.now()
	sdir = 'some_folder'
	snum = SNUM
	ids = which_haloes(snum)	

	#get hubble constant and redshift
	header = readsnap(sdir, snum, 0, header_only=1)
	h = header["hubble"]
	redshift = header["redshift"]
	a = header["time"]
	t_snapshot = age_conversion(redshift, 0.0)
	cosm = [h, redshift, a, t_snapshot]

	filename = 'merger_flux_{0}.fits'.format(SNUM)
	create_fits_file(filename, folder='flux_between_at_snap/')
	create_fits_file(filename, folder='flux_between/')

	time_step = get_time_step() # in Myr

	for i in range(20):
		settime = datetime.now()
		t_global = get_global_time(snum) # in Myr

		if t_global != 99999.0: #condition to skip snapshot	
			p, v, Rvir, Mgas_halo, Mvir = read_haloes(ids, snum, redshift, h, a) #comoving

			n_steps = how_many_time_steps(snum) #minimun 1

			snap = readsnap(sdir, snum, 0)
			inflow, outflow, Mgas = get_flux_spherical(snap, snum, p, v, t_global, n_steps, time_step, cosm)
			write_to_fits_file(redshift, ids, p, Rvir, Mgas_halo, Mvir, inflow, outflow, 
				Mgas, t_global, n_steps, time_step, filename, folder='flux_between/') 

			inflow, outflow, Mgas = flux_at_snapshot(snap, snum, p, v, cosm)
			write_to_fits_file(redshift, ids, p, Rvir, Mgas_halo, Mvir, inflow, outflow, 
				Mgas, t_snapshot, 1, time_step, filename, folder='flux_between_at_snap/') 
			
			print("Redshift (snum) done:", redshift, "(", snum, ")")

		#preparing for the next cycle
		snum -= 1
		header = readsnap(sdir, snum, 0, header_only=1)
		h = header["hubble"]
		redshift = header["redshift"]
		a = header["time"]
		t_snapshot = age_conversion(redshift, 0.0)
		cosm = [h, redshift, a, t_snapshot]
		ids = which_haloes(snum)

		print("Snum:", snum+1, ". Time:", datetime.now()-settime)

	print("Total time:", datetime.now() - begintime)


def flux_at_snapshot(snap, snum, p_halo, V_halo, cosm, n_steps=1):
	#getting current cosmic time of snapshot
	h = cosm[0]
	redshift = cosm[1]
	a = cosm[2]
	t_snapshot = cosm[3]

	N_haloes = len(p_halo[:,0])

	#Rb = which_rvir(snum)*a # it's comoving, so I turn it into physical
	Rb = numpy.full(N_haloes, 30) # it's physical

	#creating arrays
	inflow_tot = numpy.zeros([N_haloes,n_steps])
	outflow_tot = numpy.zeros([N_haloes,n_steps])
	Mgas_tot = numpy.zeros([N_haloes,n_steps])

	for j in range(N_haloes):
		#selecting particles in halo on global comoving coordinates
		p, v, m = initial_selection(snap, p_halo[j,:], V_halo[j,:], 1.3*Rb[j]/a, h, a) 

		#comoving -> physical
		for k in range(3):
			p[:,k] = p[:,k]*a
			v[:,k] = v[:,k]*a + h*100*1.e-3*p[:,k]

		r_new2 = p[:,0]*p[:,0] + p[:,1]*p[:,1] + p[:,2]*p[:,2]

		#now choose particles at the border
		r_sel, v_r, m_sel = choose_particles(r_new2, v, p, m, Rb[j], 1)
			
		#calculating flux through the shell
		coeff = 1.022e-9 # (km/s -> kpc/yr)
		size = 1.e+6 * 300*1.022e-9  # width of the shell in kpc
		sel = (v_r < 0)
		inflow_tot[j,0] = -coeff*numpy.sum(m_sel[sel] * v_r[sel])/dr
		sel = (v_r > 0)
		outflow_tot[j,0] = coeff*numpy.sum(m_sel[sel] * v_r[sel])/dr

		#calculating gass mass inside the shell
		sel = (r_new2 < Rb[j]**2)
		m_inside = m[sel]
		Mgas_tot[j,0] = numpy.sum(m_inside)

	return inflow_tot, outflow_tot, Mgas_tot


def how_many_time_steps(snum):
	data = numpy.genfromtxt("time_steps.txt", skip_header=1)
	sel = (data[:,0] == snum)
	n_steps = len(data[:,2][sel])
	return n_steps

def get_global_time(snum):
	data = numpy.genfromtxt("time_steps.txt", skip_header=1)
	sel = (data[:,0] == snum)
	time = data[:,2][sel][0]
	return time

def which_haloes(snum):
	data = numpy.genfromtxt("father_halo.txt", skip_header=1)
	sel = (data[:,0] == snum)
	ids = data[:,2][sel][:20]
	return ids

def get_time_step():
	data = numpy.genfromtxt("time_steps.txt", skip_header=1)
	sel = (data[1:,2] != 99999)
	time_step = data[1:,2][sel][0]
	return time_step

def which_rvir(snum):
	from pandas import DataFrame, read_csv
	df = read_csv('rvir_evolution.csv')
	rvir = df['Rvir'][df['snum'] == snum]
	return numpy.array(rvir)



if __name__ == "__main__":
	snum = 1200 - int(sys.argv[1])
	main(snum)

