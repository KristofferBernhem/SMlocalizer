
import java.util.ArrayList;
import java.util.Random;

public class TestData {
	public static ArrayList<Particle> generate(int nFrames, int width, int height, int perFrame, int total){
		ArrayList<Particle> newDataset = new ArrayList<Particle>();
		Random r = new Random();
		double[] X = new double[total];
		double[] Y = new double[total];
		
		for (int i = 0; i < total; i++){
			X[i] =  0.8*width*r.nextDouble();
			Y[i] =  0.8*height*r.nextDouble();
		}
		int count = 0;
		for (int frame = 1; frame <= nFrames; frame++){			
			for (int j = 0; j < perFrame; j++){
				newDataset.add( new Particle(
						X[count]  + 50* r.nextDouble() + 0.1*width, // x.
						Y[count]  + 50* r.nextDouble() + 0.1*height, // y.
						0,
						frame,						// frame.
						1,
						100*(1 + r.nextDouble()),			// sigma x.
						100*(1 + r.nextDouble()),     	// sigma y.
						0,
						1 + 5*(10*r.nextDouble()), 	// precision x.
						1 + 5*(10*r.nextDouble()), 	// precision y
						0,
						0.1 + 0.1*r.nextDouble(),  	// chi square.
						100 + 500*r.nextDouble(),	// photons.			 						
						1));
				count++;
				if (count == total){
					count = 0; //Reset.
				}
					
			}
		}
		return newDataset;
	}
}
