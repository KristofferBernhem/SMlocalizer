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
/**
 *
 * @author kristoffer.bernhem@gmail.com
 */

import java.util.ArrayList;

import ij.process.ByteProcessor;


public class autoCorrelation {
	/*
	 * Returns the correlation for a given set of arraylist of particles with targetParticles being shifted by shift[x,y].
	 */
	public static double correlation(ArrayList<Particle> referenceParticles, ArrayList<Particle> targetParticles, int[] shift)
	{

		double corr = 0;		
		for (int i = 0; i < referenceParticles.size(); i++)
		{
			for (int j = 0; j < targetParticles.size(); j++)
			{
				if (Math.abs(referenceParticles.get(i).x - targetParticles.get(j).x - shift[0]) < 5 &&
						Math.abs(referenceParticles.get(i).y - targetParticles.get(j).y - shift[1]) < 5)
				{
					corr++;
				}
			}
		}
		return corr;
	}
	
	/*
	 * Returns the correlation for a given set of arraylist of particles with targetParticles being shifted by shift[x,y,z].
	 */
	public static double correlation3D(ArrayList<Particle> referenceParticles, ArrayList<Particle> targetParticles, int[] shift)
	{

		double corr = 0;		
		for (int i = 0; i < referenceParticles.size(); i++)
		{
			for (int j = 0; j < targetParticles.size(); j++)
			{
				if (Math.abs(referenceParticles.get(i).x - targetParticles.get(j).x - shift[0]) < 5 &&
						Math.abs(referenceParticles.get(i).y - targetParticles.get(j).y - shift[1]) < 5 &&
						Math.abs(referenceParticles.get(i).z - targetParticles.get(j).z - shift[2]) < 5)
				{
					corr++;
				}
			}
		}
		return corr;
	}
	/*
	 * Maximize correlation between the two arraylist of particles by moving targetParticles. Maximal shift is maxshift[xy,z].
	 */
	public static double[] maximize(ArrayList<Particle> referenceParticles, ArrayList<Particle> targetParticles, int[] maxShift)
	{
		boolean threeD = false;
		double[] corr = {0,0,0,0};
		int i = 0;
		while (i < referenceParticles.size() && !threeD)
		{
			if (referenceParticles.get(i).z != 0)
				threeD = true;
			i++;
		}
		i = 0;
		while (i < targetParticles.size() && !threeD)
		{
			if (targetParticles.get(i).z != 0)
				threeD = true;
			i++;
		}
		
		if (!threeD) // 2D data. Significantly faster.
		{
			int[] shift = {0,0};		

			for (int xShift = -maxShift[0]; xShift <= maxShift[0]; xShift += 5) // loop over all possible shifts.
			{
				for (int yShift = -maxShift[0]; yShift <= maxShift[0]; yShift += 5)// loop over all possible shifts.
				{
					shift[0] = xShift;
					shift[1] = yShift;
					double tempCorr = correlation(referenceParticles, targetParticles,shift);// calculate correlation for this shift.
					if (tempCorr > corr[0]) // if the new correlation is better then the previous one, update array.
					{
						corr[0] = tempCorr;
						corr[1] = shift[0];
						corr[2] = shift[1];
					}
				}	
			}
		}else // 3D data.
		{
			int[] shift = {0,0,0};		

			for (int xShift = -maxShift[0]; xShift <= maxShift[0]; xShift += 5)// loop over all possible shifts.
			{
				for (int yShift = -maxShift[0]; yShift <= maxShift[0]; yShift += 5)// loop over all possible shifts.
				{
					for (int zShift = -maxShift[1]; zShift <= maxShift[1]; zShift += 10)// loop over all possible shifts.
					{
						shift[0] = xShift;
						shift[1] = yShift;
						shift[2] = zShift;
						double tempCorr = correlation3D(referenceParticles, targetParticles,shift); // calculate correlation for this shift.
						if (tempCorr > corr[0]) // if the new correlation is better then the previous one, update array.
						{
							corr[0] = tempCorr;
							corr[1] = shift[0];
							corr[2] = shift[1];
							corr[3] = shift[2];
						}
					}
				}	
			}
		}
		return corr;
	}

	
	/*
	 *  test functions. No use for actual softare.
	 */
	public static void main(String[] args) {
		final ij.ImageJ ij = new ij.ImageJ();
		ij.setVisible(true);	
		Particle Pcenter = new Particle();
		Particle Pcenter2 = new Particle();
		int width = Math.round(12800/5);
		int height = Math.round(12800/5);

		short[][] referenceIPShort  = new short[width][height];					
		ByteProcessor referenceIP = new ByteProcessor(width,height);
		Pcenter.x = (int)(width/2);
		Pcenter.y = (int)(height/2);
		Pcenter.z = -100;
		Pcenter2.x = (int)(width/2) + 100;
		Pcenter2.y = (int)(height/2) + 100;
		Pcenter2.z = 100;
		double[] drift = {0.1,0.2,0.1};
		int nFrames = 1000;
		ArrayList<Particle> referenceList = new ArrayList<Particle>();
		ArrayList<Particle> pList = new ArrayList<Particle>();
		for (int i = 0; i < nFrames; i++)
		{
			referenceIPShort[(int)(Pcenter.x+i*drift[0])][(int)(Pcenter.y+i*drift[1])] = 1;
			referenceIP.set((int)(Pcenter.x+i*drift[0]), 
					(int)(Pcenter.y+i*drift[1]), 
					referenceIP.get((int)(Pcenter.x+i*drift[0]), 
							(int)(Pcenter.y+i*drift[1])) + 10 );
			Particle P = new Particle();
			P.x = Pcenter.x+i*drift[0];
			P.y = Pcenter.y+i*drift[1];
			P.z = Pcenter.z+i*drift[2];
			P.include = 1;
			P.channel = 1;
			pList.add(P);
			referenceList.add(P);
			referenceIPShort[(int)(Pcenter2.x+i*drift[0])][(int)(Pcenter2.y+i*drift[1])] = 1;
			Particle P2 = new Particle();
			P2.x = Pcenter2.x+i*drift[0];
			P2.y = Pcenter2.y+i*drift[1];
			P2.z = Pcenter2.z+i*drift[2];
			P2.include = 1;
			P2.channel = 1;
			pList.add(P2);
			referenceList.add(P2);
		}
		
		short[][] targetIPShort  = new short[width][height];
		ByteProcessor targetIP = new ByteProcessor(width,height);
		ArrayList<Particle> targetList = new ArrayList<Particle>();
		for (int i = nFrames; i < 2*nFrames; i++)
		{
			targetIPShort[(int)(Pcenter.x+i*drift[0])][(int)(Pcenter.y+i*drift[1])] = 1;
			targetIP.set((int)(Pcenter.x+i*drift[0]), 
					(int)(Pcenter.y+i*drift[1]), 
					targetIP.get((int)(Pcenter.x+i*drift[0]), 
							(int)(Pcenter.y+i*drift[1])) + 10);
			Particle P = new Particle();
			P.x = Pcenter.x+i*drift[0];
			P.y = Pcenter.y+i*drift[1];
			P.z = Pcenter.z+i*drift[2];

			P.include = 1;
			P.channel = 2;
			pList.add(P);
			targetList.add(P);
			targetIPShort[(int)(Pcenter2.x+i*drift[0])][(int)(Pcenter2.y+i*drift[1])] = 1;
			Particle P2 = new Particle();
			P2.x = Pcenter2.x+i*drift[0];
			P2.y = Pcenter2.y+i*drift[1];
			P2.z = Pcenter2.z+i*drift[2];
			P2.include = 1;
			P2.channel = 2;
			pList.add(P2);
			targetList.add(P2);
		}
		boolean[] renderCh = new boolean[2];
		renderCh[0] = true;
		renderCh[1] = true;
		int[] pixelSize = {1,1};
		boolean gSmoothing = true;
		referenceIP.blurGaussian(2);
		targetIP.blurGaussian(2);
	//	generateImage.create("", renderCh, pList, width, height, pixelSize, gSmoothing); // show test data.



		ArrayList<Particle> pList2 = new ArrayList<Particle>();
		int[] maxShift = {250,250};
		double[] shift3 = maximize(referenceList, targetList, maxShift);	
	/*	for (int i = 0; i < nFrames; i++)
		{

			Particle P = new Particle();
			P.x = Pcenter.x+i*drift[0];
			P.y = Pcenter.y+i*drift[1];
			P.include = 1;
			P.channel = 1;
			pList2.add(P);
			Particle P2 = new Particle();
			P2.x = Pcenter2.x+i*drift[0];
			P2.y = Pcenter2.y+i*drift[1];
			P2.include = 1;
			P2.channel = 1;
			pList2.add(P2);
		}
		*/for (int i = nFrames; i < 2*nFrames; i++)
		{

			Particle P = new Particle();
			P.x = Pcenter.x+i*drift[0] + shift3[1];
			P.y = Pcenter.y+i*drift[1] + shift3[2];
			P.include = 1;
			P.channel = 2;
			pList2.add(P);
			Particle P2 = new Particle();
			P2.x = Pcenter2.x+i*drift[0]+ shift3[1];
			P2.y = Pcenter2.y+i*drift[1]+ shift3[2];
			P2.include = 1;
			P2.channel = 2;
			pList2.add(P2);
		}

		generateImage.create("", renderCh, pList2, width, height, pixelSize, gSmoothing); // show test data.
	}

}
