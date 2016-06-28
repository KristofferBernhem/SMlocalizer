import java.util.ArrayList;


public class AutoCorrelation {
	public static double[] getLambda(ArrayList<Particle> First, ArrayList<Particle> Second, int[] stepSize,int[] lb, int[] ub){		
		double[][] tempData = new double[First.size()][2];
		for (int i = 0; i < First.size(); i++){
			Particle tempParticle = First.get(i);
			tempData[i][0] = tempParticle.x;
			tempData[i][1] = tempParticle.y;
		}
		double[][]Dataset1 = tempData;	
		double[][] tempData2 = new double[Second.size()][2];
		for (int i = 0; i < Second.size(); i++){
			Particle tempParticle = Second.get(i);
			tempData2[i][0] = tempParticle.x;
			tempData2[i][1] = tempParticle.y;
		}
		double[][]Dataset2 = tempData2;

		double[] lambda = {0,0};
		double[] optimLambda = {0,0};

		double MinVal = value(lambda,Dataset1,Dataset2);		
		for (int i = lb[0]; i <= ub[0]; i+=stepSize[0]){
			for (int j = lb[1]; j <= ub[1]; j+=stepSize[1]){
				double[] lambdaTest = {i,j};
				
				double Test = value(lambdaTest,Dataset1,Dataset2);
				if (Test < MinVal){
					MinVal = Test;
					optimLambda[0] = lambdaTest[0];
					optimLambda[1] = lambdaTest[1];
				}				
			}				
		}		
		return optimLambda;
	} // end getLambda

	public static double value(double[] lambda, double[][] Dataset1, double[][] Dataset2){
		double eval = 0; // output, summed nearest neighbor distance for all particles in Dataset1 against Dataset2.

		for (int i = 0; i < Dataset1.length; i++){		// Loop over all particles in Dataset1.		
			double NN_x = (Dataset1[i][0] - Dataset2[0][0] + lambda[0])*(Dataset1[i][0] - Dataset2[0][0] + lambda[0]) +
					(Dataset1[i][1] - Dataset2[0][1] + lambda[1])*(Dataset1[i][1] - Dataset2[0][1] + lambda[1]);

			for (int j = 1; j < Dataset2.length;j++){	// Find smallest neighbor distance.
				if ((Dataset1[i][0] - Dataset2[j][0] + lambda[0])*(Dataset1[i][0] - Dataset2[j][0] + lambda[0]) + 
						(Dataset1[i][1] - Dataset2[j][1] + lambda[1])*(Dataset1[i][1] - Dataset2[j][1] + lambda[1]) 
						< NN_x){
					NN_x = (Dataset1[i][0] - Dataset2[j][0] + lambda[0])*(Dataset1[i][0] - Dataset2[j][0] + lambda[0]) +
							(Dataset1[i][1] - Dataset2[j][1] + lambda[1])*(Dataset1[i][1] - Dataset2[j][1] + lambda[1]);
				}

			}					
			eval += NN_x; // Add nearest neighbor distance for the current particle. 
		}	
		return eval;	
	} // end value
}
