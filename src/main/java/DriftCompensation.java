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
 * v1.0 2016-07-18 Kristoffer Bernhem. Calculates correlation between two groups of Particles and 
 * by shifting the second group tries to maximize the correlation to approximate shift between the 
 * two groups.
 */

import java.util.ArrayList;


public class DriftCompensation {
	
	public static float correlation(ArrayList<Particle> alpha, ArrayList<Particle> beta, float[] shift,int[] maxDistance) // Calculate correlation for the current shift (correlation, x, y and z).
		 {
		float correlation = 0;
		//shift[0] = 0;
		for(int referenceIndex = 0; referenceIndex < alpha.size(); referenceIndex++){	// Loop over all referenceParticles.								
			for (int shiftIndex = 0; shiftIndex < beta.size(); shiftIndex++){			// For each referenceParticle, find the shiftParticles that are close.
				float xDist = (float) (alpha.get(referenceIndex).x - 					// Distance in x dimension after shift.
						beta.get(shiftIndex).x - 
						shift[1]);
				xDist *= xDist;
				if(xDist<maxDistance[0]){												// If distance is small enough, check y.
					float yDist = (float) (alpha.get(referenceIndex).y - 				// Distance in y dimension after shift. 
							beta.get(shiftIndex).y - 
							shift[2]); 
					yDist *= yDist;
					if(yDist<maxDistance[1]){											// If distance is small enough, check z.
						float zDist = (float) (alpha.get(referenceIndex).z -  			// Distance in z dimension after shift.
								beta.get(shiftIndex).z - 
								shift[3]);
						zDist *= zDist;
						if(zDist<maxDistance[2]){										// If distance is small enough, calculate square distance.														
							float Distance = xDist+
									yDist+
									zDist;
								Distance = (float) Math.sqrt(Distance);
							if (Distance < 1){											// Avoid assigning infinity as value.
								correlation += 1;
							}else{
								correlation += 1/Distance;								// Score of how close the particles were.
							}							
						}	// if z distance ok, calculate correlation.
					} // if y distance ok, calculate z distance.
				}	// if x distance ok, calculate y distance.
			}	// loop over all particles in beta, calculate x distance.
		} // loop over all particles in alpha.
		return correlation;
	}
	public static float correlation2D(ArrayList<Particle> ref, ArrayList<Particle> target, float[] shift,int[] maxDistance)  // Calculate correlation for the current shift (correlation, x, y and z).
	 {
	float correlation = 0;
	for(int referenceIndex = 0; referenceIndex < ref.size(); referenceIndex++){	// Loop over all referenceParticles.								
		for (int shiftIndex = 0; shiftIndex < target.size(); shiftIndex++){		// For each referenceParticle, find the shiftParticles that are close.
			float xDist = (float) (ref.get(referenceIndex).x - 					// Distance in x dimension after shift.
					target.get(shiftIndex).x - 
					shift[1]);
			xDist *= xDist;
			if(xDist<maxDistance[0]){											// If distance is small enough, check y.
				float yDist = (float) (ref.get(referenceIndex).y - 				// Distance in y dimension after shift. 
						target.get(shiftIndex).y - 
						shift[2]); 
				yDist *= yDist;
				if(yDist<maxDistance[1]){										// If distance is small enough, check z.
																																		
					float Distance = xDist+										// If distance is small enough, calculate square distance.
							yDist;
						Distance = (float) Math.sqrt(Distance);					
					if (Distance < 1){											// Avoid assigning infinity as value.
						correlation += 1;
					}else{
						correlation += 1/Distance;								// Score of how close the particles were.
					}							
				}	// if y distance ok, calculate correlation.					
			} // if x distance ok, calculate y distance.
		}	// loop over all particles in beta, calculate x distance.
	} // loop over all particles in alpha.
	return correlation;
}

	public static float[] findDrift (ArrayList<Particle> Alpha, ArrayList<Particle> Beta, int[] maxShift, int[] maxDistance)
	{
		float[] shift 		= {0,0,0,0};					// {correlation, x, y, z}, output array,
		float[] tempShift 	= {0,0,0,0};					// {correlation, x, y, z}, temporary array, holds value within a given iteration.
		float[] lastShift 	= {0,0,0,0};					// {correlation, x, y, z}, temporary array, holds values from last round.
		float[] stepSize 	= { (float)(maxShift[0]/5.0),	// x.
								(float)(maxShift[0]/5.0),	// y.
								(float)(maxShift[1]/5.0)};	// z.
		boolean threeD 		= false;						// if we deal with 3d data.
		int index 			= 0;							// index of particles.
		while (index < Alpha.size())
		{
			if (Alpha.get(index).z != 0){					// if any z value is not 0 we are dealing with 3D data.
				threeD = true;
				index = Alpha.size();
			}
			index++;
		}
		index = 0;
		if (!threeD){
			while (index < Beta.size())
			{
				if (Beta.get(index).z != 0){					// if any z value is not 0 we are dealing with 3D data.
					threeD = true;
					index = Beta.size();
				}
				index++;
			}
		}
		if (threeD){											// if we have 3D data.
			for (int i = 0; i < 3; i++)							// final stepsize is maxShift/(5*10^2) in every dimension.
			{	
				for (tempShift[1] = lastShift[1]-5*stepSize[0]; tempShift[1]  <= lastShift[1]+5*stepSize[0]; tempShift[1]  += stepSize[0])
				{
					
					for (tempShift[2] = lastShift[2]-5*stepSize[1]; tempShift[2]  <= lastShift[2]+5*stepSize[1]; tempShift[2]  += stepSize[1])
					{
						for (tempShift[3] = lastShift[3]-5*stepSize[2]; tempShift[3]  <= lastShift[3]+5*stepSize[2]; tempShift[3]  += stepSize[2])
						{
							tempShift[0] = correlation(Alpha,Beta,tempShift,maxDistance);	// with current shift of Beta, calculate correlation.
							if (tempShift[0] > shift[0])	// if we got a better correlation with current shift of Beta.
							{
								shift[0] = tempShift[0];	// update correlation value.
								shift[1] = tempShift[1];	// update shift estimate in x.
								shift[2] = tempShift[2];	// update shift estimate in y.
								shift[3] = tempShift[3];	// update shift estimate in z.
							}
						} // z shift.
					} // y shift.
				} // x shift.				
				stepSize[0] /= 10;		// improve x stepsize.
				stepSize[1] /= 10;		// improve y stepsize.
				stepSize[2] /= 10;		// improve z stepsize.
				lastShift = shift;		// update information about this round, new center point based on this rounds best estimate.			
			} // main loop.
		} // 3D data.
		else{
			for (int i = 0; i < 3; i++)							// final stepsize is maxShift/(5*10^2) in every dimension.
			{	
				for (tempShift[1] = lastShift[1]-5*stepSize[0]; tempShift[1]  <= lastShift[1]+5*stepSize[0]; tempShift[1]  += stepSize[0])
				{
					
					for (tempShift[2] = lastShift[2]-5*stepSize[1]; tempShift[2]  <= lastShift[2]+5*stepSize[1]; tempShift[2]  += stepSize[1])
					{
					
						tempShift[0] = correlation2D(Alpha,Beta,tempShift,maxDistance);	// with current shift of Beta, calculate correlation.
						if (tempShift[0] > shift[0])	// if we got a better correlation with current shift of Beta.
						{
							shift[0] = tempShift[0];	// update correlation value.
							shift[1] = tempShift[1];	// update shift estimate in x.
							shift[2] = tempShift[2];	// update shift estimate in y.							
						}
						
					} // y shift.
				} // x shift.				
				stepSize[0] /= 10;		// improve x stepsize.
				stepSize[1] /= 10;		// improve y stepsize.				
				lastShift = shift;		// update information about this round, new center point based on this rounds best estimate.			
			} // main loop.
		} // 2D data.	
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
		ArrayList<Particle> C = new ArrayList<Particle>();
		ArrayList<Particle> D = new ArrayList<Particle>();
		double drift = 0.20;
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
			
			Particle P6 = new Particle();
			P6.x = Psecond.x - i*drift;
			P6.y = Psecond.y - i*drift;
			Particle P7 = new Particle();
			P7.x = Psecond.x + i*drift;
			P7.y = Psecond.y + i*drift;
			C.add(P6);
			D.add(P7);
		}

		int[] maxShift = {250,250,250};		// maximal shift (+/-).

		int[] maxDistance = {50*50,50*50,50*50}; // main speedup.

		long time = System.nanoTime();
		float[] shift = findDrift (A, B, maxShift,  maxDistance);
			time -= System.nanoTime();
		time *= 1E-6;
		System.out.println(shift[1] + " x " +shift[2] + " x " + shift[3] + " x "  + " yields " + shift[0]  + " in " + time + " ms.");
		time = System.nanoTime();
		shift = findDrift (C, D, maxShift,  maxDistance);
			time -= System.nanoTime();
		time *= 1E-6;
		System.out.println(shift[1] + " x " +shift[2] + " x " + shift[3] + " x "  + " yields " + shift[0]  + " in " + time + " ms.");

	}

}
