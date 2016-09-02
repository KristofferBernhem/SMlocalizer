<<<<<<< HEAD
/* Copyright 2016 Kristoffer Bernhem.
 * This file is part of SMLocalizer.
 *
 *  SMLocalizer is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SMLocalizer is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SMLocalizer.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * TODO Add center, frame, channel input and fix updated output particle.
 * Evaluate with second and third gauss description.
 */
public class GaussSolver {

	int[] data, center;
	int width, size, maxIterations, frame, pixelSize, channel;		
	double[] P;
	double totalSumOfSquares, convCritera;
	public GaussSolver(int[] data, int width, double convCriteria, int maxIterations, int[] center, int channel, int pixelSize, int frame)
	{
		this.data 			= data; 			// data to be fitted.
		this.width 			= width; 			// data window width.
		this.size 			= width*width;		// total number of datapoints.
		this.center 		= center; 			// center coordinates.
		this.channel 		= channel; 			// channel id.
		this.frame			= frame;
		this.pixelSize 		= pixelSize;		// pixelsize  in nm.
		this.convCritera 	= convCriteria; 	// convergence criteria
		this.maxIterations 	= maxIterations;	// maximal number of iterations.
		double mx 			= 0; 				// moment in x (first order).
		double my 			= 0; 				// moment in y (first order).
		double m0 			= 0; 				// 0 order moment.

		for (int i = 0; i < data.length; i++) // get image moments.
		{
			int x = i % width;
			int y = i / width; 
			mx += x*data[i];
			my += y*data[i];
			m0 += data[i];
		}
		double[] tempP = {
				data[width*(width-1)/2 + (width-1)/2], 	// use center pixel value to approximate max value.
				mx/m0,									// weighted centroid as approximation.
				my/m0,									// weighted centroid as approximation.
				width/3.5,								// approximation of sigma x.
				width/3.5,								// approximation of sigma y.
				0,										// theta.
				0};										// offset, set to 0 initially.
		this.P = tempP;									// add parameter estimate to P.

		m0 /=  size;									// Mean value.
		for (int i = 0; i < data.length; i++){			// Calculate total sum of squares for R^2 calculations.
			this.totalSumOfSquares += (data[i] -m0)*(data[i] -m0);
		}

	} // constructor.

	public static void main(String[] args) { // testcase
/*		int[] testdata ={ // slize 45 SingleBead2
				3888, 3984,  6192,   4192, 3664,  3472, 3136,
				6384, 8192,  12368, 12720, 6032,  5360, 3408, 
				6192, 13760, 21536, 20528, 9744,  6192, 2896,
				6416, 15968, 25600, 28080, 12288, 4496, 2400,
				4816, 11312, 15376, 14816, 8016,  4512, 3360,
				2944, 4688,  7168,   5648, 5824,  3456, 2912,
				2784, 3168,  4512,   4192, 3472,  2768, 2912
		};*/
		int[] testdata2 = {
				3296, 4544,  5600,  5536,  5248,  4448, 3328,
				3760, 5344,  8240,  9680, 10592,  7056, 3328,
				3744, 6672, 14256, 24224, 20256, 11136, 5248,
				3696, 7504, 16944, 26640, 21680, 10384, 5008,
				2992, 6816, 10672, 15536, 14464,  7792, 4016,
				2912, 3872,  4992,  6560,  6448,  4896, 3392,
				3088, 3248,  3552, 	3504,  4144,  4512, 2944  
		};


		int width = 7;
		int maxIterations = 1000;
		int pixelSize = 100;
		int channel = 1;
		int frame = 1;
		int[] center = {5,5};
		double convergence = 1E-5;
		long start = System.nanoTime();
		for (int i = 0; i < 100; i++){ 
			GaussSolver Gsolver = new GaussSolver(testdata2, width, convergence, maxIterations, center, channel, pixelSize,frame);		
			Gsolver.Fit();
		}
		long stop = System.nanoTime() - start;
		System.out.println(stop/1000000); 
		GaussSolver Gsolver = new GaussSolver(testdata2, width, convergence, maxIterations, center, channel, pixelSize,frame);		
		Gsolver.Fit();
		System.out.println(Gsolver.P[0] + " " + Gsolver.P[1] + " x "+ Gsolver.P[2]+ " " +Gsolver.P[3] + " x "+ Gsolver.P[4]+" " + Gsolver.P[5] + " x "+ Gsolver.P[6]);
	} // main

	public Particle Fit()
	{
		///////////////////////////////////////////////////////////////////
		////////////////////// Setup calculations: ////////////////////////
		///////////////////////////////////////////////////////////////////

		double[] bounds = {
				0.6			, 1.4,				// amplitude.
				P[1] - 2	, P[1] + 2,			// x.
				P[2] - 2	, P[2] + 2,			// y.
				width/14.0	, width / 1.5,		// sigma x.
				width/14.0	, width / 1.5,		// sigma y.
				-0.5*Math	.PI,0.5*Math.PI,	// theta.
				-0.5		, 0.5				// offset.
		};

		double[] stepSize = {
				0.1,				// amplitude.
				0.25*100/pixelSize,	// x.
				0.25*100/pixelSize,	// y
				0.5*100/pixelSize,	// sigma x.
				0.5*100/pixelSize, 	// sigma y.
				0.1965,				// theta.
				0.01				// offset.
		};
		double Rsquare 	= 1;			// start value.
		double ThetaA 	= 0;			// Initialize, used for gaussian function evaluation.
		double ThetaB 	= 0;			// Initialize, used for gaussian function evaluation.
		double ThetaC 	= 0;			// Initialize, used for gaussian function evaluation.
		double residual = 0;			// Initialize, used for gaussian function evaluation.
		double tempRsquare = 0;			// Initialize, used for gaussian function evaluation.

		// update bonds based on center pixel value.
		bounds[0] 	= bounds[0] * P[0];  	// input bounds are fractional, based on image max intensity.
		bounds[1] 	= bounds[1] * P[0];  	// input bounds are fractional, based on image max intensity.
		bounds[12] 	= bounds[12] * P[0];  	// input bounds are fractional, based on image max intensity.
		bounds[13] 	= bounds[13] * P[0];  	// input bounds are fractional, based on image max intensity.
		stepSize[0] = stepSize[0] * P[0];  	// input stepSize are fractional, based on image max intensity.
		stepSize[6] = stepSize[6] * P[0];  	// input stepSize are fractional, based on image max intensity.

		int iterationCount 	= 0; 		// keep track of number of iterations.
		double oldRsquare	= Rsquare;	// keep track of last main round of optimizations goodness of fit.
		int pId 			= 0; 		// parameter id.
		boolean optimize 	= true; 	// true whilst still optimizing parameters.

		///////////////////////////////////////////////////////////////////
		////////////////////// Optimize parameters:////////////////////////
		///////////////////////////////////////////////////////////////////

		
		while (optimize)
		{						
			if (pId == 0) // if we start a new full iteration over all parameters.
				oldRsquare = Rsquare; // store the last iterations goodness of fit.
			if (P[pId] + stepSize[pId] > bounds[pId*2] && P[pId] + stepSize[pId] < bounds[pId*2 + 1]){
				P[pId] += stepSize[pId]; // add stepSize to the current parameter and evaluate gaussian.
				ThetaA = Math.cos(P[5]) * Math.cos(P[5]) / (2 * P[3]*P[3]) + 
						Math.sin(P[5]) * Math.sin(P[5]) / (2 * P[4]*P[4]);
				ThetaB = -Math.sin(2 * P[5]) / (4 * P[3]*P[3]) + 
						Math.sin(2 * P[5]) / (4 * P[4]*P[4]);
				ThetaC = Math.sin(P[5]) * Math.sin(P[5]) / (2 * P[3]*P[3]) + 
						Math.cos(P[5]) * Math.cos(P[5]) / (2 * P[4]*P[4]);

				tempRsquare = 0; // reset.
				for (int xyIndex = 0; xyIndex < width * width; xyIndex++)
				{
					int xi = xyIndex % width;
					int yi = xyIndex / width;
					residual = P[0] * Math.exp(-(ThetaA * (xi -  P[1]) * (xi -  P[1]) -
							2 * ThetaB * (xi -  P[1]) * (yi - P[2]) +
							ThetaC * (yi - P[2]) * (yi - P[2])
							)) + P[6] - data[xyIndex];
					tempRsquare += residual * residual;
				}

				tempRsquare = (tempRsquare / totalSumOfSquares);  // normalize.

				if (tempRsquare < Rsquare)                // If improved, update variables.
				{				
					Rsquare = tempRsquare;
				} // update parameters
				else // if there was no improvement
				{
					P[pId] -= stepSize[pId]; // reset the current parameter.
					if (stepSize[pId] < 0)  // if stepSize is negative, try positive direction at reduced stepSize.
						stepSize[pId] /= -1.5;
					else 					// if stepSize is positive, try changing direction.
						stepSize[pId] /= -1;
				}
			}
			else // if stepSize is out of bounds.
				stepSize[pId] /= 2; // reduce stepSize.
			pId++; // update parameter id.
			if (pId > 6){ // if all parameters has been evaluated this round.
				pId = 0; // reset.
				if (iterationCount > 12){ // if two rounds has been run.
					if ((oldRsquare  - Rsquare) < convCritera) // check if we improved the fit this full round by atleast the convergence criteria.
					{	
						optimize = false;	// exit.
					}				
				}
			}
			iterationCount++; // update iteration count.
			if(iterationCount > maxIterations) // if we reached max iterations.
				optimize = false; // exit
		} // optimize loop.


		ThetaA = Math.cos(P[5]) * Math.cos(P[5]) / (2 * P[3]*P[3]) + 
				Math.sin(P[5]) * Math.sin(P[5]) / (2 * P[4]*P[4]);
		ThetaB = -Math.sin(2 * P[5]) / (4 * P[3]*P[3]) + 
				Math.sin(2 * P[5]) / (4 * P[4]*P[4]);
		ThetaC = Math.sin(P[5]) * Math.sin(P[5]) / (2 * P[3]*P[3]) + 
				Math.cos(P[5]) * Math.cos(P[5]) / (2 * P[4]*P[4]);

		tempRsquare = 0; // reset.
		for (int xyIndex = 0; xyIndex < width * width; xyIndex++)
		{
			int xi = xyIndex % width;
			int yi = xyIndex / width;
			tempRsquare = P[0] * Math.exp(-(ThetaA * (xi -  P[1]) * (xi -  P[1]) -
					2 * ThetaB * (xi -  P[1]) * (yi - P[2]) +
					ThetaC * (yi - P[2]) * (yi - P[2])
					)) + P[6];			 
		}

		///////////////////////////////////////////////////////////////////
		///////////////////////// Final output: ///////////////////////////
		///////////////////////////////////////////////////////////////////

		Particle Localized 		= new Particle();
		Localized.include 		= 1;
		Localized.channel 		= channel;
		Localized.frame   		= frame;
		Localized.r_square 		= 1-Rsquare;
		Localized.x				= pixelSize*(P[1] + center[0] - Math.round((width)/2));
		Localized.y				= pixelSize*(P[2] + center[1] - Math.round((width)/2));
		Localized.z				= pixelSize*0;	// no 3D information.
		Localized.sigma_x		= pixelSize*P[3];
		Localized.sigma_y		= pixelSize*P[4];
		Localized.sigma_z		= pixelSize*0; // no 3D information.
		Localized.photons		= tempRsquare;
		Localized.precision_x 	= Localized.sigma_x/Localized.photons;
		Localized.precision_y 	= Localized.sigma_y/Localized.photons;
		Localized.precision_z 	= Localized.sigma_z/Localized.photons;		
		return Localized;
	}

public class GaussSolver {

	int[] data;
	int width;	
	int size;
	double[] P;
	public GaussSolver(int[] data, int width)
	{
		this.data 	= data;
		this.width 	= width;
		this.size 	= width*width; 
		double mx = 0; // moment in x (first order).
		double my = 0; // moment in y (first order).
		double m0 = 0; // 0 order moment.

		for (int i = 0; i < data.length; i++)
		{
			int x = i % width;
			int y = i / width; 
			mx += x*data[i];
			my += y*data[i];
			m0 += data[i];
		}
		double[] tempP = {data[width*(width-1)/2 + (width-1)/2],
				mx/m0,
				my/m0,
				width/3.5,
				width/3.5,
				0,
				0}; // calculate startpoints:
		this.P = tempP;
	}
	
	public static void main(String[] args) { // testcase
		int[] testdata ={ // slize 45 SingleBead2
				3888, 3984,  6192,   4192, 3664,  3472, 3136,
				6384, 8192,  12368, 12720, 6032,  5360, 3408, 
				6192, 13760, 21536, 20528, 9744,  6192, 2896,
				6416, 15968, 25600, 28080, 12288, 4496, 2400,
				4816, 11312, 15376, 14816, 8016,  4512, 3360,
				2944, 4688,  7168,   5648, 5824,  3456, 2912,
				2784, 3168,  4512,   4192, 3472,  2768, 2912
		};
		
		int width = 7;
		GaussSolver Gsolver = new GaussSolver(testdata, width);
		double[] delta = {1,1};
		Gsolver.Gauss(delta);
	}

	public double[][] Jacobian(double[] delta) // jacobian description of gauss.
	{
		double[][] J = new double[size][7];
		 
		return J;

	}
	
	public double Gauss(double[] delta) // function evaluation.
	{
		double G = 0;
		/*
		 * P[0]: Amplitude
		 * P[1]: x0
		 * P[2]: y0
		 * P[3]: sigma x
		 * P[4]: sigma y
		 * P[5]: theta
		 * P[6]: offset 
		 */		
		double ThetaA = Math.cos(P[5])*Math.cos(P[5])/(2*P[3]*P[3]) + Math.sin(P[5])*Math.sin(P[5])/(2*P[4]*P[4]); 
		double ThetaB = -Math.sin(2*P[5])/(4*P[3]*P[3]) + Math.sin(2*P[5])/(4*P[4]*P[4]); 
		double ThetaC = Math.sin(P[5])*Math.sin(P[5])/(2*P[3]*P[3]) + Math.cos(P[5])*Math.cos(P[5])/(2*P[4]*P[4]);


		for (int i = 0; i < size; i++){
			int xi = i % width;
			int yi = i / width;	
			G += P[0]*Math.exp(-(ThetaA*(xi - P[1])*(xi - P[1]) - 
					2*ThetaB*(xi - P[1])*(yi - P[2]) +
					ThetaC*(yi - P[2])*(yi - P[2])
					)) + P[6];

		}
		return G;
		
	}

>>>>>>> refs/remotes/origin/master
}
