import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class autoCorr3 {
	ArrayList<Particle> 
	referenceParticle,		// Reference, check the other list against this one.
	shiftParticle;			// Shift this one to maximize correlation between the two lists.
	//double[] stepSize;				// x, y and z step size in shift calculations. 
	int[] maxShift;				// Maximal shift to calculate.
	int[] maxDistance;
	double convergence;
	int maxIterations;

	autoCorr3(ArrayList<Particle> Alpha, ArrayList<Particle> Beta, int[] maxShift, int[] maxDistance, double convergence, int maxIterations){
		this.referenceParticle 	= Alpha;
		this.shiftParticle 		= Beta;
		//this.stepSize 			= stepSize;
		this.maxShift 			= maxShift;
		this.maxDistance 		= maxDistance; //max distance in any 1 dimension.
		this.convergence		= convergence;
		this.maxIterations		= maxIterations;
	}



	public double Correlation(double[] shift){  // Calculate correlation for the current shift (correlation, x, y and z).
		double Corr = 0;

		for(int referenceIndex = 0; referenceIndex < referenceParticle.size(); referenceIndex++){				// Loop over all referenceParticles.								
			for (int shiftIndex = 0; shiftIndex < shiftParticle.size(); shiftIndex++){			// For each referenceParticle, find the shiftParticles that are close.
				double xDist = (referenceParticle.get(referenceIndex).x - 					// Distance in x dimension after shift.
						shiftParticle.get(shiftIndex).x - 
						shift[1]);
				xDist *= xDist;
				if(xDist<maxDistance[0]){												// If distance is small enough, check y.
					double yDist = (referenceParticle.get(referenceIndex).y - 				// Distance in y dimension after shift. 
							shiftParticle.get(shiftIndex).y - 
							shift[2]); 
					yDist *= yDist;
					if(yDist<maxDistance[1]){											// If distance is small enough, check z.
						double zDist = (referenceParticle.get(referenceIndex).z -  			// Distance in z dimension after shift.
								shiftParticle.get(shiftIndex).z - 
								shift[3]);
						zDist *= zDist;
						if(zDist<maxDistance[2]){										// If distance is small enough, calculate square distance.														
							double Distance = xDist+
									yDist+
									zDist;
							//	Distance = Math.sqrt(Distance);
							if (Distance < 1){										// Avoid assigning infinity as value.
								Corr += 1;
							}else{
								Corr += 1/Distance;									// Score of how close the particles were.
							}							
						}	
					}
				}
			}
		}
		return Corr;
	}

	public double[] optimize3D()
	{		
		double[] correlation = {0,0,0,0}; // correlation, x, y and z shift to aquire that correlation.
		double[] tempCorrelation = {0,0,0,0}; // correlation, x, y and z shift to aquire that correlation.
		// get a rough estimate of shift by scanning the entire range set by boundry in a corse stepsize. This avoids getting stuck at local minima.
		for (double xShift = (double) - maxShift[0]; xShift <= (double) maxShift[0]; xShift += (double) maxShift[0]/5.0)
		{
			tempCorrelation[1] = xShift;
			for (double yShift = (double) - maxShift[1]; yShift <= (double) maxShift[1]; yShift += (double) maxShift[1]/5.0)
			{
				tempCorrelation[2] = yShift;
				for (double zShift = (double) - maxShift[2]; zShift <= (double) maxShift[2]; zShift += (double) maxShift[2]/5.0)
				{
					tempCorrelation[3] = zShift;

					tempCorrelation[0] = Correlation(tempCorrelation);

					if (tempCorrelation[0]  > correlation[0])
					{	

						correlation[0] = tempCorrelation[0];			
						correlation[1] = tempCorrelation[1];
						correlation[2] = tempCorrelation[2];
						correlation[3] = tempCorrelation[3];

					}
				}
			}
		}
		// with the local area in parameter space found, locate the exact shift iteratively-

		boolean optimize = true;
		int iterationCount = 0;
		double[] stepSize = {maxShift[0]/5,maxShift[0]/5,maxShift[0]/5};
		//stepSize[0] = (double) maxShift[0]/5;
		//stepSize[1] = (double) maxShift[1]/5;
		//stepSize[2] = (double) maxShift[2]/5;
		correlation[0] = 0;

		double[] lastRoundCorr = correlation;
		while(optimize)
		{

			lastRoundCorr = correlation;

			for (double xShift = lastRoundCorr[1] - stepSize[0]; xShift <= lastRoundCorr[1] + stepSize[0]; xShift += stepSize[0])
			{
				tempCorrelation[1] = xShift;
				for (double yShift = lastRoundCorr[2] - stepSize[1]; yShift <= lastRoundCorr[2] + stepSize[1]; yShift += stepSize[1])
				{
					tempCorrelation[2] = yShift;
					for (double zShift = lastRoundCorr[3] - stepSize[2]; zShift <= lastRoundCorr[3] + stepSize[2]; zShift += stepSize[2])
					{
						tempCorrelation[3] = zShift;

						tempCorrelation[0] = Correlation(tempCorrelation);

						if (tempCorrelation[0]  > correlation[0])
						{	
							correlation[0] = tempCorrelation[0];			
							correlation[1] = tempCorrelation[1];
							correlation[2] = tempCorrelation[2];
							correlation[3] = tempCorrelation[3];

						}
					}
				}
			}
			for (int id = 1; id < 4; id++)
			{
				if (correlation[id] == lastRoundCorr[id]) // if there was step taken in this dimension.
				{
					stepSize[id-1] /= 1.2; 				// decrease stepsize.
				}
			}

			iterationCount++;

			if (iterationCount > 20) // might take a few rounds of decreasing stepsize before improvement is found.
			{
				if ((correlation[0] - lastRoundCorr[0]) / correlation[0] < convergence) // demand on continued improvement.
					optimize  = false;
			}


			if (iterationCount > maxIterations)
				optimize = false; // exit
		}


		//	System.out.println(correlation[0] + " yields: " + correlation[1] + " x " +correlation[2] + " x " +correlation[3]);
		return correlation;
	}

	public static double[] findDrift (ArrayList<Particle> Alpha, ArrayList<Particle> Beta, int[] maxShift, int[] maxDistance, double convergence, int maxIterations)
	{
		autoCorr3 AC = new autoCorr3(Alpha,Beta,maxShift,maxDistance, convergence, maxIterations);
		double[] shift = AC.optimize3D();	
		return shift;
	}
	public static void main(String[] args){ // test case.
		Particle P = new Particle();
		P.x = 100;
		P.y = 100;
		P.z = 50;		
		Particle Psecond = new Particle();
		Psecond.x = 1000;
		Psecond.y = 1000;
		Psecond.z = 500;	
		ArrayList<Particle> A = new ArrayList<Particle>();
		ArrayList<Particle> B = new ArrayList<Particle>();
		double drift = 0.3;
		for (double i = 0; i < 200; i++){
			Particle P2 = new Particle();
			P2.x = P.x - i*drift;
			P2.y = P.y - i*drift;
			P2.z = P.z - 2*i*drift;

			A.add(P2);
			Particle P3 = new Particle();
			P3.x = P.x + i*drift;
			P3.y = P.y + i*drift;
			P3.z = P.z + 2*i*drift;

			B.add(P3);

			Particle P4 = new Particle();
			P4.x = Psecond.x - i*drift;
			P4.y = Psecond.y - i*drift;
			P4.z = Psecond.z - 2*i*drift;

			A.add(P4);
			Particle P5 = new Particle();
			P5.x = Psecond.x + i*drift;
			P5.y = Psecond.y + i*drift;
			P5.z = Psecond.z + 2*i*drift;

			B.add(P5);			
		}

		int[] maxShift = {250,250,250};		// maximal shift (+/-).

		int[] maxDistance = {50*50,50*50,50*50}; // main speedup.
		int maxIteration = 1000;
		double convergence = 1E-3;

		int n = 50;

		long start = System.nanoTime();
		//	for (int i = 0; i < n; i++)
		//{
		//	findDrift(A,B,stepSize,maxShift,maxDistance,convergence,maxIteration);
		//	}
		int processors 					= Runtime.getRuntime().availableProcessors();				// Number of processor cores on this system.
		ExecutorService exec 			= Executors.newFixedThreadPool(processors);					// Set up parallel computing using all cores.
		List<Callable<double[]>> tasks 	= new ArrayList<Callable<double[]>>();						// Preallocate.

		for (int i = 0; i < n; i++){
			Callable<double[]> c = new Callable<double[]>() {													// Computation to be done.
				@Override
				public double[] call() throws Exception {		
					return findDrift(A,B,maxShift,maxDistance,convergence,maxIteration);																			// Actual call for each parallel process.
				}
			};
			tasks.add(c);																					// Que this task.
		}

		double[][] shift = new double[3][n];
		try {
			List<Future<double[]>> parallelComputeSmall = exec.invokeAll(tasks);		// Execute computation.
			double[] Corr;

			for (int i = 0; i < parallelComputeSmall.size(); i++){							// Loop over and transfer results.
				try {
					Corr = parallelComputeSmall.get(i).get();							
					shift[0][i] = Corr[1];											// Update best guess at x shift.
					shift[1][i] = Corr[2];											// Update best guess at y shift.
					shift[2][i] = Corr[3];											// Update best guess at z shift.

				} catch (ExecutionException e) {
					e.printStackTrace();
				}
			}
		} catch (InterruptedException e) {

			e.printStackTrace();
		}
		finally {
			exec.shutdown();	// Shut down connection to cores.
		}	



		long stop = System.nanoTime();
		long elapsed = (stop-start)/1000000;

		System.out.println(n + " rounds in " + elapsed + " ms");





	}

}
