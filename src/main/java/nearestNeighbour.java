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
 * nearestNeighbour analysis not currently active in SMLocalizer v 1.5 implementation pending.
 * 
 */
import java.util.ArrayList;
import java.util.Random;

public class nearestNeighbour {

	public static double[] analyse()
	{
		ArrayList<Particle> results = TableIO.Load();
		int[] startIdx = new int[results.get(results.size()-1).channel];
		int currIdx = 0;
		startIdx[0]  = 0; // first comparion  starts at index 0.
		for (int i = 0; i < results.size(); i++)
		{
			if (results.get(i).channel > results.get(startIdx[currIdx]).channel &&
					results.get(i).channel  > 1)
			{				

				currIdx++;					
				startIdx[currIdx] = i;	

			}				
		} // obtain which groups to compare with.
		double[] nnResults = new double[25]; // 10 nm steps with final bin being for larger distances.
		int binSize = 10;
		// this loops over ch1 and compares to ch2 only.
		for (int referenceIdx = startIdx[1]; referenceIdx < results.size()-1;referenceIdx++ )
			//for (int referenceIdx = startIdx[0]; referenceIdx < startIdx[1]-1; referenceIdx++)
		{
			if (results.get(referenceIdx).include == 1)
			{
				double distance = 10000;
				for (int targetIdx = startIdx[0]; targetIdx < startIdx[1]-1; targetIdx++)
					//for (int targetIdx = startIdx[1]; targetIdx < results.size()-1;targetIdx++ )
				{
					if (results.get(targetIdx).include==1)
					{

						double tempDistance = Math.sqrt((results.get(referenceIdx).x-results.get(targetIdx).x)*(results.get(referenceIdx).x-results.get(targetIdx).x) +
								(results.get(referenceIdx).y-results.get(targetIdx).y)*(results.get(referenceIdx).y-results.get(targetIdx).y) + 
								(results.get(referenceIdx).z-results.get(targetIdx).z)*(results.get(referenceIdx).z-results.get(targetIdx).z)
								);
						if (tempDistance < distance)
							distance = tempDistance;
					}
				}
				boolean sorted = false;
				int bin = 1;			
				while (!sorted)
				{
					if (distance < (double)bin*binSize)
					{
						nnResults[bin-1]++;
						sorted = true;
					}
					bin++;
					if (bin == 26) // exit
					{
						//nnResults[24]++;
						sorted = true;
					}

				}

			}		
		} // nnResults now populated.
		// display nnResults:
		int[] x = {0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250};
		correctDrift.plot(nnResults, x);

		//for (int i = 0; i < nnResults.length;i++)
		//	nnResults[i] /= (startIdx[1] - startIdx[0]);
		//correctDrift.plot(nnResults, x);
		return nnResults;
	} // end analyse.

	public static void main(final String... args) 
	{
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);

		int referencePcount = 1000; // number of particles to be generated in reference set-
		int targetPcount = 300;// number of particles to be generated in target set-
		double maxX = 12800;
		double maxY = 12800;
		double[] resultV = new double[25];
		for (int repeat = 0; repeat < 1000; repeat++)
		{
			Random r = new Random();
			ArrayList<Particle> testData = new ArrayList<Particle>();
			for (int i = 0; i < referencePcount; i++)
			{
				Particle P = new Particle();
				P.channel = 1; // reference channel
				P.z = 0;
				P.x = r.nextDouble() * maxX;
				P.y = r.nextDouble() * maxY;
				P.include = 1;
				testData.add(P);
			}
			for (int i = 0; i < targetPcount; i++)
			{
				Particle P = new Particle();
				P.channel = 2; // reference channel
				P.z = 0;
				P.x = r.nextDouble() * maxX;
				P.y = r.nextDouble() * maxY;
				P.include = 1;
				testData.add(P);
			}
			TableIO.Store(testData);
			double[] temp = analyse();
			for (int j = 0; j < temp.length; j++)
			{
				resultV[j] += temp[j];
			}
		}


		int[] x = {0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250};
		correctDrift.plot(resultV, x);
	} // end main

}
