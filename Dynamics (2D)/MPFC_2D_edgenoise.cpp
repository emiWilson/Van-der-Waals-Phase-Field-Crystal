#include <iostream>
#include <sstream>
#include <cmath>
#include <fstream>
#include <complex>
#include <ctime>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <fftw3-mpi.h>
#define Pi (2.*acos(0.))
using namespace std;

//MPI parameters
int argc;
char **argv;
int myid,numprocs;
ptrdiff_t alloc_local, local_size, local_start;

ptrdiff_t Nx_kt2,Nx_k;					//needed for padding real arrays for transform

//domain parameters
int Nx,Ny;					//number of grid points in x and y
double dx,dy;				//Grid spacing
double lambdalatt;				//typical expected interatomic distance
unsigned int ptsperLambda;			//points per expected interatomic distance
double facx,facy;			//fourier factors for scaling lengths in k-space

//time parameters
int totalTime;		//simulation time and print frequency
double dt;						//numerical time step
int restartFlag,restartTime;	//flag for restarting (set to 1) and time

//miscellaneous 
int run_id;			//id given as executable argument
int timeseed, seed;	//seed based on local time and final code seed

//*************initial condition and simulation variables

double no;			//average density
double ns,nl;		//average solid and liquid densities (for initialization)

double noiseAmp;		//noise amplitude
double alpha;
double tau,Bx;		//tau (static case) and Bx

//*************Measured global quantities
double avgn0;		//averga density per grid point
double N;			//Total "mass" in the system
double F;			//Averaged free energy
double MU;			//Averaged chemical potential
double P;			//averaged pressure
double Pcoord;
int coord[2];

double a1, b, d;
double b1,b2, b3, b4, b5, b6, b7, b8, b9, b10;


//************* Parameters to linearly change Ptarget
double t_Ptarget_i,t_Ptarget_f;
double Ptarget_i,Ptarget_f;
int FlagPIntermediate;
double t_Ptarget_int, Ptarget_int;

//************* Stuff related to determining print times
int numPrints,tnextprint,tfirst,PrintCounter;
int printfreq;

//Initialization parameters
int InitPattern,InitRadius;	// initialization pattern and characteristic radius

//*******************Arrays for fields
//real space arrays
double *n;		//density and fourier transform of time derivative of density
double *n_NL;		//non-linear terms for density and concentration
double *n_NL_conv;

double *k_sqr;
//complex arrays
complex<double> *kn;			//k-space density and time derivative
complex<double> *kn_NL;		//k-space non-linear terms
complex<double> *kn_NL_conv;


double beta, lambda;		//calc_nmf wavelength for output

double a2, a3, a4;

double Tm;
double M;

double A, B, C, DB;

//calc_nmf
double *nmf;
complex<double> *knmf;
fftw_plan planF, planB;

//fftw plans
fftw_plan planF_n, planB_n;	//forward and backward plans for density
fftw_plan planF_NL_n, planF_NL_n_conv;		//non-linear forward transforms

fftw_plan planB_nmf;

//*******************Strings & streams
string configpath;	//name of the config file
string datafoldername;	//name of the data output folder
string npre;		//name of n file prefix (ex: in n0.dat, the 'n')


ofstream f_stream;	//stream for output data

//______________________________________________Function headers
void freeMem();
void timeStepn();

void calcn_NL();
void calcn_NL_conv();
void outputField(int time,double *Array,string filepre);


void setfftwPlans();
void fftwMPIsetup();
void allocateArrays();
void domainParams();
void readConfig();
void initn();
void normalize(double* Array);

void seedrng(int argc, char* argv[]);

void calc_nmf(double* Arrayin,double* Arrayout,double lambda);
void calc_k2();


//______________________________________________main
int main(int argc, char* argv[])
{	
	clock_t begin = clock();
	timeseed=time(NULL);

	int t;
	configpath="ConfigMPFC_2D.txt";
	npre="n_";

	//Read config file
	readConfig();
	/*a1 = 4.65;
	b = 0.5;
	Tm = 1;
	d = 3.3;

	a2 = 2.0+ 6.0 * tau + 0.17*tau*tau;
	a3 = -1.2- 0.6*tau +0.14*tau*tau;
	a4 = 0.11;
	*/
	A = 50;
	B = -19;
	C = 50;

	a2 = tau + 0.7;
	a3 = -1./2.;
	a4 = 1./3.;

	b2 = tau + 0.7;
	b3 = 50 * tau - 19 -1/2;
	b4 = 50 + 1/3;

/*
	b1 = (2. *tau - a1);
	b2 = 0.0;
	b3 = 0.666667* tau;
	b4 = 0.0;
	b5 = 0.4 * tau;
	b6 = 0.0;
	b7 = 0.285714* tau;
	b8 = 0.0;
	b9 = 0.222222* tau;
	b10 = 0.0 * tau;
	*/

	M = 1.;

	//set up mpi environment
	fftwMPIsetup();
	MPI_Barrier(MPI_COMM_WORLD);
	
	//seed random number generator depending on executable options
	seedrng(argc, argv);
	
	//set up domain parameters
	domainParams();

	//allocate real and k-space arrays
	allocateArrays();

	//set up fftw plans
	setfftwPlans();

	//set up initial conditons
	//initialize(argc,argv);
	initn();
    MPI_Barrier(MPI_COMM_WORLD);
    outputField(0,n,npre);
	
	MPI_Barrier(MPI_COMM_WORLD);
	calc_k2();
	MPI_Barrier(MPI_COMM_WORLD);

	//calc and output nmf, comment these two lines if don't want to see what nmf looks like
	fftw_execute(planF_n);

	MPI_Barrier(MPI_COMM_WORLD);
	calc_nmf(n,nmf,lambda);

	MPI_Barrier(MPI_COMM_WORLD);
	fftw_execute(planB_nmf);
	normalize(nmf);
	//outputField(0, nmf, "nmf");
	//tau = 0.1;
	for(t=restartTime+1;t<totalTime+1;t++)
	{	
		MPI_Barrier(MPI_COMM_WORLD);
		fftw_execute(planF_n);

		MPI_Barrier(MPI_COMM_WORLD);
		calc_nmf(n,nmf,lambda);

		MPI_Barrier(MPI_COMM_WORLD);
		fftw_execute(planB_nmf);
		normalize(nmf);
		
		//Deal with non-linear terms
		calcn_NL();					//calculate the non-linear terms
		fftw_execute(planF_NL_n);	//forward transform the collection of non-linear terms
		
		timeStepn();				//numerically integerate
		
		fftw_execute(planB_n);		//back transform density
		normalize(n);				//normalize transform
		
		MPI_Barrier(MPI_COMM_WORLD);

		if( t%printfreq == 0 )		//Output of global quantities
			{
				if (myid==0) 
					
					printf("time = %d, temp = %f\n",t,tau);

					//print density fields
                	outputField(t,n,npre);
                	
                	//outputField(t,nmf, "nmf");
					
			}


	}
	MPI_Barrier(MPI_COMM_WORLD);
	//free system memory
	freeMem();
	
	MPI_Finalize();

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout<< myid << ": Done, Elapsed time is "<< elapsed_secs <<" seconds." <<endl;
	
	return 0;
}

//____________________________________________________Function bodies
//____________________________________________________Function bodies
void timeStepn()
{
	ptrdiff_t i,j;
	ptrdiff_t index,index1;
	ptrdiff_t yj;
	
	double kx,ky,k2, ks;
	double prefactor_n;
	//double count = 1;
	double LinTerms;
	double k_val;
	double qo = 1;//2*Pi/lambdalatt;
	double chi;

	for(j=0;j<local_size;j++)
	{
		yj = j + local_start;
		if ( yj < Ny/2 )
			ky = yj*facy;
		else
			ky = (yj-Ny)*facy;
		
		index1 = j*Nx_k;
		for(i=0;i<Nx_k;i++)
		{
			index = i + index1;
			kx = i*facx;
			
			k2 = kx/dx*kx/dx + ky/dy*ky/dy;
			ks = k_sqr[index];
			k_val = sqrt(k2);
			chi = exp(-k2/(2*lambda*lambda));
			//k2 = -0.5*( (3*cos(kx)-cos(ky)+(cos(kx)*cos(ky)-3) )/(dx*dx)+(3*cos(ky)-cos(kx)+(cos(kx)*cos(ky)-3) )/(dy*dy) );
			
			LinTerms = a2 *(1 - chi) + b2 * chi + Bx*(-2*k2 + k2 *k2);//-a2 + a2*(1 - chi) + b2 * chi + tau + Bx * ( 1  - exp(-0.5 * (k_val - qo) * (k_val - qo))); 
			//LinTerms = ( (tau+Bx)  -  Bx * exp(-0.5 * (k_val - qo) * (k_val - qo))) ;//( tau+Bx*(1-2*k2+k2*k2));
			prefactor_n =1./( 1. + dt*alpha*k2*LinTerms);
				
			kn[index] = prefactor_n * ( kn[index] - dt*alpha*k2*(kn_NL[index] + n_NL_conv[index] * chi));
			//if (k2 != ks)
			//{
				//printf("%lf %lf %lf %lf %lf %lf %lf \n", prefactor_n, dt, alpha, k2, tau, Bx, LinTerms);
			//	printf("%lf %lf \n", k2, ks);
			//	count = count + 1;
			//}	
			
		}
	}
}
void calcn_NL()
{
	ptrdiff_t i,j;
	ptrdiff_t index,index1;
	
	for(j=0;j<local_size;j++)
	{
		index1 = j*Nx_kt2;
		for(i=0;i<Nx;i++)
		{
			index = i + index1;

			//n_NL[index] = -n[index]*n[index]/2.+n[index]*n[index]*n[index]/3. + 1./3. * (A*tau+B)*nmf[index]*nmf[index]+1./4.*C*nmf[index]*nmf[index]*nmf[index];
		
			n_NL[index] = a3 * (n[index]*n[index] - 1./3. * nmf[index]*nmf[index]) + a4 *(n[index]*n[index]*n[index] - 1./4. * nmf[index]*nmf[index]*nmf[index])
							+ 1./3. * (b3)*nmf[index]*nmf[index] + 1./4.*(b4)*nmf[index]*nmf[index]*nmf[index];
							//	+ 1./3. * (b3)*nmf[index]*nmf[index]+1./4.*(b4)*nmf[index]*nmf[index]*nmf[index];

			//a3 * (n[index]*n[index] - nmf[index]*nmf[index]) + a4 *(n[index]*n[index]*n[index] - nmf[index]*nmf[index]*nmf[index])
							//	+ 1./3. * (b3)*nmf[index]*nmf[index]+1./4.*(b4)*nmf[index]*nmf[index]*nmf[index];
		}
	}	
}
void calcn_NL_conv()
{
	ptrdiff_t i,j;
	ptrdiff_t index,index1;
	
	for(j=0;j<local_size;j++)
	{
		index1 = j*Nx_kt2;
		for(i=0;i<Nx;i++)
		{
			index = i + index1;

			n_NL_conv[index] = -2./3.*(a3)*n[index]*nmf[index] - 3./4.*(b4) *n[index]*nmf[index]*nmf[index]
								+ 2./3.*(b3)*n[index]*nmf[index]+ 3./4.*(b4) *n[index]*nmf[index]*nmf[index];
			//2./3.*(b3)*nmf[index]*nmf[index]+ 3./4.*(b4) *nmf[index]*nmf[index]*nmf[index];
		
		}		
	}	
}

void normalize(double *Array)
{
	ptrdiff_t i,j;
	ptrdiff_t index,index1;
	
	for(j=0;j<local_size;j++)
	{
		index1 = j*Nx_kt2;
		for(i=0;i<Nx;i++)
		{
			index = i + index1;
			
			Array[index] /= Nx*Ny;
		}
	}
}
void calc_k2()
{
	ptrdiff_t i,j;
	ptrdiff_t index, index1;
	ptrdiff_t yj;

	double kx, ky;

	for(j=0;j<local_size;j++)
	{
		yj = j + local_start;

		if ( yj < Ny/2 )
			ky =yj*facy/dy;
		else
			ky = (yj-Ny)*facy/dy;

		index1 = j*Nx_k;
		for(i=0;i<Nx_k;i++)
		{
			index = i + index1;

			kx = i*facx/dx;
	
			k_sqr[index] = kx*kx + ky*ky;//( 3. - cos(kx) - cos(ky) - cos(kx)*cos(ky) )/dx/dx;
		}
	}
}
void calc_nmf(double* Arrayin,double* Arrayout,double lambda)
{
	ptrdiff_t i,j;
	ptrdiff_t index;
	double chi;
	
	for(j=0;j<local_size;j++)
	{
		for(i=0;i<Nx_k;i++)
		{
			index = i + j*Nx_k;;

			chi = exp(-k_sqr[index]/(2*lambda*lambda));
			knmf[index]=kn[index]*chi;
		}
	}	
}
void freeMem()
{
	free(n);

	free(n_NL);
	free(n_NL_conv);
	
	fftw_free(kn);

	fftw_free(kn_NL);
	fftw_free(kn_NL_conv);
	
	fftw_destroy_plan(planF_n);
	fftw_destroy_plan(planB_n);
	fftw_destroy_plan(planF_NL_n);
	
	free(nmf);
	free(knmf);
	fftw_destroy_plan(planF);
	fftw_destroy_plan(planB);
}
void outputField(int time,double *Array,string filepre)
{	
	ptrdiff_t i,j;
	string filename;
	std::stringstream sstm;
	
    sstm << datafoldername << filepre << time <<".dat";
	filename = sstm.str();
	
	int kk;
	
	for (kk=0;kk<numprocs;kk++)
	{
		if (myid == kk)
		{
			if (myid==0)
			{
				f_stream.open (filename.c_str());
				if(!f_stream.is_open())
				{
					printf("Unable to open file for writing\n");
					printf("Exiting simulation!\n");
					exit(1);
				}
			}
			else
			{
				f_stream.open (filename.c_str(),std::fstream::app);
				if( !f_stream.is_open())
				{
					printf("Unable to open file for writing\n");
					printf("Exiting simulation!\n");
					exit(1);
				}
			}
			
			for(j=0;j<local_size;j++)
			{
				for(i=0;i<Nx;i++)
				{
					f_stream << Array[i+j*Nx_kt2]<<"\n ";
				}
			}
			f_stream.close();
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
}
void setfftwPlans()
{
	//n
	planF_n = fftw_mpi_plan_dft_r2c_2d(Ny, Nx, n, reinterpret_cast<fftw_complex*>(kn), MPI_COMM_WORLD, FFTW_MEASURE);
	planB_n = fftw_mpi_plan_dft_c2r_2d(Ny, Nx, reinterpret_cast<fftw_complex*>(kn), n, MPI_COMM_WORLD, FFTW_MEASURE);

	planF_NL_n = fftw_mpi_plan_dft_r2c_2d(Ny, Nx, n_NL, reinterpret_cast<fftw_complex*>(kn_NL), MPI_COMM_WORLD, FFTW_MEASURE);
	planF_NL_n_conv = fftw_mpi_plan_dft_r2c_2d(Ny, Nx, n_NL_conv, reinterpret_cast<fftw_complex*>(kn_NL_conv), MPI_COMM_WORLD, FFTW_MEASURE);

	
	planB_nmf = fftw_mpi_plan_dft_c2r_2d(Ny, Nx, reinterpret_cast<fftw_complex*>(knmf), nmf, MPI_COMM_WORLD, FFTW_MEASURE);
	
	//calc_nmf
	planF = fftw_mpi_plan_dft_r2c_2d(Ny, Nx, n,  reinterpret_cast<fftw_complex*>(knmf), MPI_COMM_WORLD, FFTW_MEASURE);
	planB = fftw_mpi_plan_dft_c2r_2d(Ny, Nx,  reinterpret_cast<fftw_complex*>(knmf), n, MPI_COMM_WORLD, FFTW_MEASURE);
	
}
void allocateArrays()
{
	//n
	n = (double*) fftw_malloc( sizeof (double*) * 2*alloc_local );
	n_NL = (double*) fftw_malloc( sizeof (double*) * 2*alloc_local );
	n_NL_conv = (double*) fftw_malloc( sizeof (double*) * 2*alloc_local );

	kn = new complex<double>[alloc_local];
    kn_NL = new complex<double>[alloc_local];
	kn_NL_conv = new complex<double>[alloc_local];	

	//Measured quantities
	nmf = (double*) fftw_malloc( sizeof (double*) * 2*alloc_local );
	knmf = new complex<double>[alloc_local];

	k_sqr = (double*) fftw_malloc( sizeof (double*) * alloc_local );
	
}
void fftwMPIsetup()
{
	//starting mpi daemons
	MPI_Init(&argc,&argv); 
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	
	printf("\n myid: %d of %d processors\n",myid,numprocs);
	
	fftw_mpi_init();
	
	//get the local (for current cpu) array sizes
	alloc_local = fftw_mpi_local_size_2d(Ny, Nx/2+1, MPI_COMM_WORLD, &local_size, &local_start);
	
	printf(" %d: local_size=%zu local_start=%zu\n",myid,local_size,local_start);
}
void domainParams()
{	
	//Determine grid properties
	dx=lambdalatt/ptsperLambda;
	dy=lambdalatt/ptsperLambda;
	
	Nx_k = (Nx/2+1);
	Nx_kt2 = 2*Nx_k;
	
	//set up fourier scaling factors
	facx = 2.*Pi/(Nx);
	facy = 2.*Pi/(Ny);
	
}

void readConfig()
{
	string line;
	ifstream in(configpath.c_str());
	
	getline(in,line);
	getline(in,line);
	sscanf (line.c_str(), "%d %d", &Nx,&Ny);
	getline(in,line);
	sscanf (line.c_str(), "%lf %d ",&lambdalatt,&ptsperLambda);
	getline(in,line);
	sscanf (line.c_str(), "%lf %d %d",&dt,&totalTime, &printfreq);
	getline(in,line);
	sscanf (line.c_str(), "%d %d",&InitPattern,&InitRadius);
	getline(in,line);
	sscanf (line.c_str(), "%lf %lf %lf",&nl,&no,&ns);
	getline(in,line);
	sscanf (line.c_str(), "%lf %lf",&Bx,&tau);
	getline(in,line);
	sscanf (line.c_str(), "%lf",&alpha);
	getline(in,line);
	sscanf (line.c_str(), "%lf",&lambda);
}
void initn()
{
	ptrdiff_t i,j;
	ptrdiff_t index,index1;
	ptrdiff_t yj;
    
	
	double atri = -4.5;	//amplitude for mode expansion ( +/- depending on side of phase diagram )
    double qo=1.;     //wavelength for triangle
	double fs;
	double avgns=0,avgnslocal=0;
	double scounter=0,scounterlocal=0;
	double rVal;

	double yy, y_max, xx, x_max, yM, ym, size;
	
	switch (InitPattern) {
		case 0:
			if (myid==0)
				cout<<"Initialization: uniform field"<<endl;
			
			for(j=0;j<local_size;j++)
			{
				yj = j + local_start;
				
				index1 = j*Nx_kt2;
				for(i=0;i<Nx;i++)
				{
					index = i + index1;

					n[index] = no;
				}
			}
			break;
			
		case 1:		//Random noise
			if (myid==0)
				cout<<"Initialization: Random noise"<<endl;
			
			for(j=0;j<local_size;j++)
			{
				yj = j + local_start;
				
				index1 = j*Nx_kt2;
				for(i=0;i<Nx;i++)
				{
					index = i + index1;
					n[index] = no+2.*(2.*drand48()-1.);
				}
			}
			break;
			
		case 2:
			if (myid==0)
				cout<<"Initialization: uniform 1-mode approx"<<endl;
			
			for(j=0;j<local_size;j++)
			{
				yj = j + local_start;
				
				index1 = j*Nx_kt2;
				for(i=0;i<Nx;i++)
				{
					index = i + index1;

						//triangle
						n[index] = no + atri * ( cos(qo*dx*i)*cos(qo/sqrt(3.)*dx*yj) - 0.5*cos(2./sqrt(3.)*qo*dx*yj) );
						avgnslocal+=n[index];
						//stripe
						//n[index] = no - 1 *  cos(qo*dx*i);
						//square
						//n[index] = no + 2.*(0.2*(cos(qo*i*dx)+ cos(qo*j*dx)) + 0.125*(cos(qo*(i+j)*dx) + cos(qo*(i-j)*dx)));
				}
			}
			MPI_Reduce(&avgnslocal, &avgns, 1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
			if (myid==0)
				avgns/=(Nx*Ny);		//determine the actual solid density inserted (varies from ns due to edges)
			MPI_Bcast(&avgns,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
			
			for(j=0;j<local_size;j++)
			{
				yj = j + local_start;
				
				index1 = j*Nx_kt2;
				for(i=0;i<Nx;i++)
				{
					index = i + index1;
					//triangle
					n[index] = n[index]*no/avgns;
				}
			}
			break;

		case 3:
			if (myid==0)
				cout<<"Initialization: 1-mode approx slab form nl, n0 and ns"<<endl;
			
			//InitRadius=Ny/2*(no-nl)/(ns-nl);
			if (myid==0) cout<<"Initial radius: "<<InitRadius<<endl;
			
			for(j=0;j<local_size;j++)
			{
				yj = j + local_start;
				
				index1 = j*Nx_kt2;
				for(i=0;i<Nx;i++)
				{
					index = i + index1;

					if( yj < (Ny/2+InitRadius) && yj > (Ny/2-InitRadius) ) 
					{
						//triangle
						n[index] = ns + atri * ( cos(qo*dx*i)*cos(qo/sqrt(3.)*dx*yj) - 0.5*cos(2./sqrt(3.)*qo*dx*yj) );
					}else{
						/*if (yj < (Ny/2+InitRadius + 40) && yj > (Ny/2-InitRadius - 40) ){
						rVal = abs(yj - Ny/2) - InitRadius;

						n[index] = nl+ atri * ( cos(qo*dx*i)*cos(qo/sqrt(3.)*dx*yj) - 0.5*cos(2./sqrt(3.)*qo*dx*yj) )*exp(-rVal/4);
						}else{*/
							n[index] = no;
						/*}*/
					}
				}
			}

			break;
			 
		case 4:
			if (myid==0)
				cout<<"Initialization: 1-mode approx circle"<<endl;
			
			for(j=0;j<local_size;j++)
			{
				yj = j + local_start;
				
				index1 = j*Nx_kt2;
				for(i=0;i<Nx;i++)
				{
					index = i + index1;
					if ( sqrt( (yj-Ny/2)*(yj-Ny/2) + (i-Nx/2)*(i-Nx/2) ) <= InitRadius )
					{
						//triangle
						rVal = sqrt( (yj-Ny/2)*(yj-Ny/2) + (i-Nx/2)*(i-Nx/2) );
						n[index] = (ns + atri * ( cos(qo*dx*i)*cos(qo/sqrt(3.)*dx*yj) - 0.5*cos(2./sqrt(3.)*qo*dx*yj) ));
						avgnslocal+=n[index];
						scounterlocal=scounterlocal+1;
						//stripe
						//n[index] = no - 1 *  cos(qo*dx*i);
						//square
						//n[index] = no + 2.*(0.2*(cos(qo*i*dx)+ cos(qo*yj*dx)) + 0.125*(cos(qo*(i+yj)*dx) + cos(qo*(i-yj)*dx)));
					}
					else
					{
						rVal = sqrt( (yj-Ny/2)*(yj-Ny/2) + (i-Nx/2)*(i-Nx/2) ) - InitRadius;
						n[index] = nl;// + atri*( cos(qo*dx*i)*cos(qo/sqrt(3.)*dx*yj) - 0.5*cos(2./sqrt(3.)*qo*dx*yj) )*(exp(-rVal/(2 * 3 * 3)));
						//n[index] = nl;// + 0.1*(2.*drand48()-1.);	//fill the rest of the box with liquid at the right density
					}
				}
				
			}
			break;

			case 5:

			y_max = (Nx - 1)*dx;
			x_max = (Ny - 1)*dx;
			size = 4.5 * 10.5;

			if (myid==0)
				cout<<"Initialization: 1-mode approx hexagonal seed"<<endl;

			for(j=0;j<local_size;j++)
			{
				yj = j + local_start;
				yy = (j + local_start)*dx - y_max/2.0;

				index1 = j*Nx_kt2;
				for(i=0;i<Nx;i++)
				{

					xx = (i*dx) - x_max/2.0;
					index = i + index1;

					n[index] = nl;

					if( xx>=-size && xx<=size )
					{
						if( (xx<-size/2.0)||(xx>size/2.0) )
						{
							yM =(size-fabs(xx))*sqrt(3.0);
							ym = -yM;
						}else{
							ym = -size*sqrt(3.)/2.;
							yM = size*sqrt(3.)/2.;
						}

						if( yy>=ym && yy<=yM )
						{
							//n[index] = (ns + atri * ( cos(qo*dx*i)*cos(qo/sqrt(3.)*dx*yj) - 0.5*cos(2./sqrt(3.)*qo*dx*yj) ));
							n[index] = ns + atri*( cos(qo*xx+Pi)*cos(qo*yy/sqrt(3.)) - .5*cos(2.*qo*yy/sqrt(3.)) );
							avgnslocal+=n[index];
							scounterlocal=scounterlocal+1;
							//stripe
							//n[index] = no - 1 *  cos(qo*dx*i);
							//square
							//n[index] = no + 2.*(0.2*(cos(qo*i*dx)+ cos(qo*yj*dx)) + 0.125*(cos(qo*(i+yj)*dx) + cos(qo*(i-yj)*dx)));
						}	
					}

				}
			}
			break;

			case 6:
			if (myid==0)
				cout<<"Initialization: Random noise, circle seed"<<endl;
			
			for(j=0;j<local_size;j++)
			{
				yj = j + local_start;
				
				index1 = j*Nx_kt2;
				for(i=0;i<Nx;i++)
				{
					index = i + index1;
					
					if ( sqrt( (yj-Ny/2)*(yj-Ny/2) + (i-Nx/2)*(i-Nx/2) ) <= InitRadius )
					{
						n[index] = no+1.5*(2.*drand48()-1.);					
					}
					else
					{
						n[index] = no;	//fill the rest of the box with liquid at the right density
					}
				}
			}
			break;
		
	}

}

void seedrng(int argc, char* argv[]) {
	
	if (argc==1) {
		cout<<"Seeding with current time and mpi id"<<endl;		
		seed=timeseed+myid;
	}
	if (argc==2) {
		cout<<"Seeding with given random number seed and mpi id"<<endl;
		seed=atoi(argv[1])+myid;
	}
	if (argc==3) {
		run_id=atoi(argv[2]);
		cout<<"Run_id: "<<run_id<<endl;
		cout<<"Seeding with given seed, mpi id and run id"<<endl;
		seed=atoi(argv[1])+myid+numprocs*run_id;
	}
		
	cout<<myid<<": seed: "<<seed<<endl;
	srand48(seed);
}

