import org.apache.commons.math3.analysis.MultivariateMatrixFunction;
import org.apache.commons.math3.analysis.MultivariateVectorFunction;

/*
 * X - lamda_x
 * sum((Xx-Yx - lamda_x)^2 + (Xy-Yy - lamda_y)^2
 */

public class CorrelationFcn {
	
	double[][] Dataset1;
	double[][] Dataset2;
	int size;
	
	public CorrelationFcn(double[][] Inp1, double[][] Inp2){
		this.Dataset1 = Inp1; // x and y coordinates for first data set.
		this.Dataset2 = Inp2; // x and y coordinates for second data set.
		this.size 	  = Dataset1.length;
	}
	
	public MultivariateVectorFunction retMVF (){
		return new MultivariateVectorFunction() {
			//@Override 
			@Override
			public double[] value(double[] lamda ){
			//		throws IllegalArgumentException {
			
				double[] eval = new double[1];
				
				for (int i = 0; i < Dataset1.length; i++){
					//double[] NearestN_x = new double[Dataset2.length];
					double NN_x = (Dataset1[i][0] - Dataset2[0][0] + lamda[0])*(Dataset1[i][0] - Dataset2[0][0] + lamda[0]);
					double NN_y = (Dataset1[i][0] - Dataset2[0][0] + lamda[0])*(Dataset1[i][0] - Dataset2[0][0] + lamda[0]);
					//double[] NearestN_y = new double[Dataset2.length];
					for (int j = 1; j < Dataset2.length;j++){						
						if ((Dataset1[i][0] - Dataset2[j][0] + lamda[0])*(Dataset1[i][0] - Dataset2[j][0] + lamda[0]) < NN_x){
							NN_x = (Dataset1[i][0] - Dataset2[j][0] + lamda[0])*(Dataset1[i][0] - Dataset2[j][0] + lamda[0]);
						}
													
						if ((Dataset1[i][1] - Dataset2[j][1] + lamda[1])*(Dataset1[i][1] - Dataset2[j][1] + lamda[1]) < NN_y){
							NN_y = (Dataset1[i][1] - Dataset2[j][1] + lamda[1])*(Dataset1[i][1] - Dataset2[j][1] + lamda[1]);
						}
					}					
					eval[0] += NN_x*NN_x + NN_y*NN_y; // Add nearest neighbor distance for 
					
				}
		
				return eval;

			}
		};
	}
	
	public MultivariateMatrixFunction retMMF() {
		return new MultivariateMatrixFunction() {

		//	@Override
			@Override
			public double[][] value(double[] point)
					throws IllegalArgumentException {
				System.out.println("her");
				return jacobian(point);
			}

			private double[][] jacobian(double[] lamda) {
				double[][] jacobian = new double[size][2];
				/*
				 * f(x,y) = sum[ (Dataset1_x-Dataset2_x + lamda_x)^2 + (Dataset1_y-Dataset2_y + lamda_y)], for the pair of Dataset1 and 2 that has closest distance.
				 * f(x,y)/d(lamda_x) = 2Dataset1_x-2Dataset2_x+2lamda_x
				 * f(x,y)/d(lamda_y) = 2Dataset1_y-2Dataset2_y+2lamda_y
				 */
				
				for (int i = 0; i < Dataset1.length; i++){
					//double[] NearestN_x = new double[Dataset2.length];
					double NN_x = 2*Dataset1[i][0] - 2*Dataset2[0][0] + 2*lamda[0];
					double NN_y = 2*Dataset1[i][1] - 2*Dataset2[0][1] + 2*lamda[1];
					//double[] NearestN_y = new double[Dataset2.length];
					for (int j = 1; j < Dataset2.length;j++){						
						if (2*Dataset1[i][0] - 2*Dataset2[0][0] + 2*lamda[0] < NN_x){
							NN_x =  2*Dataset1[i][0] - 2*Dataset2[0][0] + 2*lamda[0];
						}
													
						if (2*Dataset1[i][1] - 2*Dataset2[0][1] + 2*lamda[1] < NN_y){
							NN_y = 2*Dataset1[i][1] - 2*Dataset2[0][1] + 2*lamda[1];
						}
					}					
					 
					jacobian[i][0] = NN_x;
					jacobian[i][1] = NN_y;
				}
				
				return jacobian;
			}
		};
	}


	
}
