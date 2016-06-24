package sm_localizer;

import java.util.ArrayList;


public class AutoCorrelation {
	
//	double[][] Dataset1;
//	double[][] Dataset2;

	public static double[] getLamda(ArrayList<Particle> First, ArrayList<Particle> Second, int[] stepSize,int[] lb, int[] ub){
		double[][] tempData = new double[First.size()][2];
		for (int i = 0; i < First.size(); i++){
			Particle tempParticle = First.get(i);
			tempData[i][0] = tempParticle.x;
			tempData[i][1] = tempParticle.y;
		}
		double[][]Dataset1 = tempData;
	//	this.Dataset1 = tempData;
		double[][] tempData2 = new double[First.size()][2];
		for (int i = 0; i < First.size(); i++){
			Particle tempParticle = Second.get(i);
			tempData2[i][0] = tempParticle.x;
			tempData2[i][1] = tempParticle.y;
		}
		double[][]Dataset2 = tempData2;

		double[] lamda = {0,0};
		double[] optimLamda = {0,0};

		double MinVal = value(lamda,Dataset1,Dataset2);		
		for (int i = lb[0]; i <= ub[0]; i+=stepSize[0]){
			for (int j = lb[1]; j <= ub[1]; j+=stepSize[1]){
				double[] lamdaTest = {i,j};
				
				double Test = value(lamdaTest,Dataset1,Dataset2);
				if (Test < MinVal){
					MinVal = Test;
					optimLamda[0] = lamdaTest[0];
					optimLamda[1] = lamdaTest[1];
				}				
			}				
		}		
		return optimLamda;
	}


	public static double value(double[] lamda, double[][] Dataset1, double[][] Dataset2){
		//		throws IllegalArgumentException {

		double eval = 0;

		for (int i = 0; i < Dataset1.length; i++){
			//double[] NearestN_x = new double[Dataset2.length];
			double NN_x = (Dataset1[i][0] - Dataset2[0][0] + lamda[0])*(Dataset1[i][0] - Dataset2[0][0] + lamda[0]) +
					(Dataset1[i][1] - Dataset2[0][1] + lamda[1])*(Dataset1[i][1] - Dataset2[0][1] + lamda[1]);
			//double NN_y = (Dataset1[i][1] - Dataset2[0][1] + lamda[1])*(Dataset1[i][1] - Dataset2[0][1] + lamda[1]);
			//double[] NearestN_y = new double[Dataset2.length];
			for (int j = 1; j < Dataset2.length;j++){						
				if ((Dataset1[i][0] - Dataset2[j][0] + lamda[0])*(Dataset1[i][0] - Dataset2[j][0] + lamda[0]) + 
						(Dataset1[i][1] - Dataset2[j][1] + lamda[1])*(Dataset1[i][1] - Dataset2[j][1] + lamda[1]) 
						< NN_x){
					NN_x = (Dataset1[i][0] - Dataset2[j][0] + lamda[0])*(Dataset1[i][0] - Dataset2[j][0] + lamda[0]) +
							(Dataset1[i][1] - Dataset2[j][1] + lamda[1])*(Dataset1[i][1] - Dataset2[j][1] + lamda[1]);
				}

			}					
			eval += NN_x; // Add nearest neighbor distance for 
		}
		
/*		for (int i = 0; i < Dataset1.length; i++){
			//double[] NearestN_x = new double[Dataset2.length];
			double NN_x = (Dataset1[i][0] - Dataset2[0][0] + lamda[0])*(Dataset1[i][0] - Dataset2[0][0] + lamda[0]);
			double NN_y = (Dataset1[i][1] - Dataset2[0][1] + lamda[1])*(Dataset1[i][1] - Dataset2[0][1] + lamda[1]);
			//double[] NearestN_y = new double[Dataset2.length];
			for (int j = 1; j < Dataset2.length;j++){						
				if ((Dataset1[i][0] - Dataset2[j][0] + lamda[0])*(Dataset1[i][0] - Dataset2[j][0] + lamda[0]) < NN_x){
					NN_x = (Dataset1[i][0] - Dataset2[j][0] + lamda[0])*(Dataset1[i][0] - Dataset2[j][0] + lamda[0]);
				}

				if ((Dataset1[i][1] - Dataset2[j][1] + lamda[1])*(Dataset1[i][1] - Dataset2[j][1] + lamda[1]) < NN_y){
					NN_y = (Dataset1[i][1] - Dataset2[j][1] + lamda[1])*(Dataset1[i][1] - Dataset2[j][1] + lamda[1]);
				}
			}					
			eval += NN_x*NN_x + NN_y*NN_y; // Add nearest neighbor distance for 
		}*/
		return eval;	
	}
}
