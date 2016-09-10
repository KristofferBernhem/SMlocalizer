import java.util.ArrayList;

import net.imagej.ImageJ;

public class driftCorr {
	ArrayList<Particle> 
	referenceParticle,		// Reference, check the other list against this one.
	shiftParticle;			// Shift this one to maximize correlation between the two lists.
	int[] maxShift;				// Maximal shift to calculate.

	driftCorr(ArrayList<Particle> Alpha, ArrayList<Particle> Beta, int[] maxShift){
		this.referenceParticle 	= Alpha;
		this.shiftParticle 		= Beta;
		this.maxShift			= maxShift;
	
	}
	
	/*
	 * IDEA: calculate NN distance vector. Filter out large numbers outside of maxShift. Mean or median of these shifts should maximize correlation.
	 */
/*	public double[] optimize()
	{
//		double[] shiftX = new double[referenceParticle.size()];
//		double[] shiftY = new double[referenceParticle.size()];
//		double[] shiftZ = new double[referenceParticle.size()];
	/*	double minDist  = maxShift[0];
		double distance = 0;
		double x,y,z;
		int idx = shiftParticle.size();
		int entry = 0;
		double[] shift = new double[3];
		for (int A = 0; A < referenceParticle.size(); A++ ) // loop over all particles.
		{
			for (int B = 0; B < shiftParticle.size(); B++)
			{
				x = (referenceParticle.get(A).x - shiftParticle.get(B).x); 
				y = (referenceParticle.get(A).y - shiftParticle.get(B).y);
				z =	(referenceParticle.get(A).z - shiftParticle.get(B).z);
				x *= x;
				y *= y;
				z *= z;
				distance = x+y+z;					
				if (distance < minDist && distance < maxShift[0])
				{
					minDist  = distance;
					idx = B;
				}
			
			}
			if (idx < shiftParticle.size())
			{
	//			shiftX[0] += (referenceParticle.get(A).x - shiftParticle.get(idx).x);
//				shiftY[1] += (referenceParticle.get(A).y - shiftParticle.get(idx).y);
//				shiftZ[2] += (referenceParticle.get(A).z - shiftParticle.get(idx).z);
				shift[0] += (referenceParticle.get(A).x - shiftParticle.get(idx).x) * (referenceParticle.get(A).x - shiftParticle.get(idx).x);
				shift[1] += (referenceParticle.get(A).y - shiftParticle.get(idx).y) * (referenceParticle.get(A).y - shiftParticle.get(idx).y);
				shift[2] += (referenceParticle.get(A).z - shiftParticle.get(idx).z) * (referenceParticle.get(A).z - shiftParticle.get(idx).z);
				
				
				entry++;
			}
			/*else
			{
				shiftX[0] += maxShift[0];
				shiftY[1] += maxShift[1];
				shiftZ[2] += maxShift[2];
				
			}*/
/*				
			idx = shiftParticle.size();
		}
		
		shift[0] /= entry;
		shift[1] /= entry;
		shift[2] /= entry;
/*		BackgroundCorrection.quickSort(shiftX, 0, shiftX.length-1);
		BackgroundCorrection.quickSort(shiftY, 0, shiftY.length-1);
		BackgroundCorrection.quickSort(shiftZ, 0, shiftZ.length-1);
		if (entry %2 != 0)
		{
			double[] outShift = {shiftX[(entry-1)/2],
					shiftY[(entry-1)/2],
					shiftZ[(entry-1)/2]};
			return outShift;
		}else
		{
			double[] outShift = {(shiftX[entry/2] + shiftX[entry/2 - 1]) / 2,
					(shiftY[entry/2] + shiftY[entry/2 - 1]) / 2,
					(shiftZ[entry/2] + shiftZ[entry/2 - 1]) / 2};
			return outShift;
		}

		return shift;
	}
	
	public static void main(String[] args){
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
		Particle P = new Particle();
		P.x = 100;
		P.y = 100;
		P.z = 50;
		ArrayList<Particle> A = new ArrayList<Particle>();
		ArrayList<Particle> B = new ArrayList<Particle>();
		double drift = 0.1;
		for (double i = 0; i < 500; i++){
			Particle P2 = new Particle();
			P2.x = P.x - i*drift;
			P2.y = P.y - i*drift;
			P2.z = P.z - i*drift;
			
			A.add(P2);
			Particle P3 = new Particle();
			P3.x = P.x + i*drift;
			P3.y = P.y + i*drift;
			P3.z = P.z + i*drift;
								
			if (i == 250)
			{
				System.out.println("A:" + P2.x);
				System.out.println("B:" + P3.x);
			}
			B.add(P3);
		}
	
		
//int width = 6400;
//int height = 6400;

/*		ByteProcessor IP  = new ByteProcessor(width,height);			
		for (int x = 0; x < width; x++){
			for (int y = 0; y < height; y++){
				IP.putPixel(x, y, 0); // Set all data points to 0 as start.
			}
			
		}
		for (int i = 0; i < coordA.length; i += 2)
		{
			IP.putPixel((int) Math.round(coordA[i]),(int) Math.round(coordA[i+1]), IP.get((int)Math.round(coordA[i]), (int)Math.round(coordA[i+1])) + 1);
			Particle P = new Particle();
			P.x	= coordA[i];
			P.y	= coordA[i+1];
			A.add(P);

		}
		for (int i = 0; i < coordB.length; i += 2)
		{
			IP.putPixel((int) Math.round(coordB[i]),(int) Math.round(coordB[i+1]), IP.get((int)Math.round(coordB[i]), (int)Math.round(coordB[i+1])) + 1);
			Particle P = new Particle();
			P.x	= coordB[i];
			P.y	= coordB[i+1];
			B.add(P);
		}		
		ImagePlus Image = new ImagePlus("Start",IP);
		Image.setImage(Image);
		Image.show(); 						*/
	/*	int[] maxShift = {500*500,250*250,250*250};	// maximal shift (+/-).
		long start = System.nanoTime();
		driftCorr AC = new driftCorr(A,B,maxShift);

		double shift[] = AC.optimize();
		long stop = System.nanoTime();
		long elapsed = (stop-start)/1000000;
		System.out.println(shift[0]+  "x" +shift[1]+"x"+shift[2] + " in " + elapsed + " ms");
	/*	ByteProcessor IP2  = new ByteProcessor(width,height);			
		for (int x = 0; x < width; x++){
			for (int y = 0; y < height; y++){
				IP2.putPixel(x, y, 0); // Set all data points to 0 as start.
			}
			
		}
	
		for (int i = 0; i < 200; i++){

				int x = (int) Math.round(A.get(i).x);
				int y = (int) Math.round(A.get(i).y);				
				IP2.putPixel(x, y, (IP2.get(x, y) + 1));
				x = (int) Math.round(B.get(i).x + shift[0]);
				y = (int) Math.round(B.get(i).y + shift[1]);				
				IP2.putPixel(x, y, (IP2.get(x, y) + 1));

		}		
		ImagePlus Image2 = new ImagePlus("Corrected",IP2);
		Image2.setImage(Image2);
		Image2.show(); 	
		
		
		
	} */
}


