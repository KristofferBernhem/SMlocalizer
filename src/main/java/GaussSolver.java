
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

}
