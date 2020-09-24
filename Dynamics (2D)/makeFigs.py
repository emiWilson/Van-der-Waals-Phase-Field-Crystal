import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#need to put some sort of thing in to let setting k-lim, maybe just another fcn. 
def makeFig(filename):
    mydata = pd.read_csv(filename, sep = "\n", header = None)
    N = int(np.sqrt(len(mydata)))
    return mydata.values.reshape(N,N)

def plot1(filename, saveAs = "temp.png"):
	plt.imshow(makeFig(filename), cmap = 'gray')
	plt.colorbar()
	plt.savefig(saveAs)
	plt.clf()

def plot3(filename1 = "k2_0.dat", filename2 = "Ck_0.dat", filename3 = "Ck_0.dat", saveAs = "initialCond.png", klim = 30):
	fig, ax = plt.subplots(1, 3, figsize=(21, 7))
	fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1)

	im1 = ax[0].imshow(makeFig(filename1), cmap = "hot")
	ax[0].set_title(filename1);
	fig.colorbar(im1, ax = ax[0]);


	im2 = ax[1].imshow(makeFig(filename2), cmap = 'hot')
	ax[1].set_xlim(0,klim)
	ax[1].set_ylim(0,klim)
	ax[1].set_title(filename2);
	fig.colorbar(im2, ax = ax[1]);

	im3 = ax[2].imshow(makeFig(filename3), cmap = 'hot')
	ax[2].set_xlim(0,klim)
	ax[2].set_ylim(0,klim)
	ax[2].set_title(filename3);
	fig.colorbar(im3, ax=ax[ 2]);

	fig.savefig(saveAs);

def plotTimestep(i = "0"):
	k_lim =30;
	fig, ax = plt.subplots(2, 3, figsize=(21, 14))
	fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1)

	im00 = ax[0,0].imshow(makeFig("n_" + str(i) + ".dat"))
	ax[0,0].set_title("n_" + str(i) );
	fig.colorbar(im00, ax = ax[0,0]);
    
	im01 = ax[0,1].imshow(makeFig("nmf_" + str(i) + ".dat"))
	ax[0,1].set_title("nmf_" + str(i) );
	fig.colorbar(im01, ax = ax[0,1]);

 #   im02 = ax[0,2].imshow(makeFig("knfNL_" + str(time) + ".dat"))
 #   ax[0,2].set_xlim(0,k_lim)
 #   ax[0,2].set_ylim(0,k_lim)
 #   ax[0,2].set_title("knfNL_" + str(i));
 #   fig.colorbar(im02, ax=ax[0,2]);

	im10 = ax[1,0].imshow(makeFig("kn_" + str(i) + ".dat"))
	ax[1,0].set_xlim(0,k_lim)
	ax[1,0].set_ylim(0,k_lim)
	ax[1,0].set_title("kn_" + str(i) );
	fig.colorbar(im10, ax = ax[1,0]);
    
	im11 = ax[1,1].imshow(makeFig("knmf_" + str(i) + ".dat"))
	ax[1,1].set_xlim(0,k_lim)
	ax[1,1].set_ylim(0,k_lim)
	ax[1,1].set_title("knmf_" + str(i) );
	fig.colorbar(im11, ax = ax[1,1]);
    
  #  im12 = ax[1,2].imshow(makeFig("knfNL_conv_" + str(time) + ".dat"))
  # ax[1,2].set_xlim(0,k_lim)
  #  ax[1,2].set_ylim(0,k_lim)
  #  ax[1,2].set_title("knfNL_" + str(i));
  #  fig.colorbar(im02, ax=ax[1,2]);
	fig.savefig(str(i) + ".png");

def makeAllFigs(filename, prefix):
	#read in Total_timesteps and timestep from Gaby8.txt
	f = open(filename) 
	raw_data = f.readlines()
	data = [] #make empty array to store data in
	#read in data as list
	line3 = raw_data[3].split(" ")
	Total_timesteps = int (line3[1])
	timeStep = int (line3[2])  
	print(str(Total_timesteps) + ", "  +  str(timeStep));
	for t0 in range(0,Total_timesteps + timeStep, timeStep):
		filename = prefix + str(t0) + ".dat"
		saveMeAs = prefix + str(t0) + ".png"
		plot1(filename, saveMeAs);

def plot3(filename1 = "n_0.dat", filename2 = "n_1000.dat", filename3 = "n_500000.dat", saveAs = "solid-liquid.png", klim = 30):

	fig, ax = plt.subplots(1, 3, figsize=(30, 10))
	fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.1)

	im = ax[0].imshow(makeFig(filename1), cmap = "gray", vmin=-2, vmax=8.5)
	ax[0].set_title("t = 0", fontsize=30);
	ax[0].axis('off')


	im = ax[1].imshow(makeFig(filename2), cmap = 'gray' , vmin = -2, vmax = 8.5)
	ax[1].set_title("t = 1000", fontsize=30);
	ax[1].axis('off')

	im = ax[2].imshow(makeFig(filename3), cmap = 'gray', vmin = -2, vmax = 8.5)
	ax[2].set_title("t = 500000", fontsize=30);
	ax[2].axis('off')

	#cb = fig.colorbar(im, ax =ax.ravel().tolist(), fraction=0.035, pad=0.03);
	#cb.set_label(label='a label',weight='bold')
	#cb.ax.tick_params(labelsize=20)


	fig.savefig(saveAs);
#plot1("n_0.dat", "n_0.png");
#plot1("n_100.dat", "n_100.png");
#plot1("n_200.dat", "n_200.png");
#plot1("n_300.dat", "n_300.png");
#plot1("n_400.dat", "n_400.png");
#plot1("n_500.dat", "n_500.png");
makeAllFigs("ConfigMPFC_2D.txt", "n_")
#makeAllFigs("ConfigMPFC_2D.txt", "nmf")
#plot3();
