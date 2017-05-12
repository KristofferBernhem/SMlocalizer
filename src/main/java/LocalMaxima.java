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

import ij.process.ImageProcessor;

public class LocalMaxima {

	public static ArrayList<int[]> FindMaxima(ImageProcessor IP, int Window, int MinLevel, int minPosPixels){		
		ArrayList<int[]> Results = new ArrayList<int[]>();
		// calculate mean value for the array:
		double mean = 0;
		int count = 0;

		int columns = IP.getWidth();
		int rows = IP.getHeight();
		int[] data = new int[columns*rows];
		for(int i = 0; i < columns*rows;i++) // loop over X then Y.
		{
			data[i] = IP.get(i);

		}	

		if (MinLevel == 0){
			for(int i = 0; i < columns*rows;i++) // loop over X then Y.
			{

				if (data[i]> 0)
					{
						mean += IP.get(i);
						count++;
					}			
					
			}	
			double std = 0;
			mean /= count;
			for (int i = 0; i < columns*rows;i++)
				std += (data[i] - mean)*(data[i] - mean);
			std /= count;
			std = Math.sqrt(std);
			MinLevel = (int) (2*mean + 3*std); // set minlevel to 
		}
		int minNeighbourValue = (int) (0.3 * MinLevel);	
		int i = (Window / 2) * columns + Window / 2; // start windowWidth / 2 pixels in and windowWidth / 2 down.
		int j = 0;
		int k = 0;
		int loopC = 0;
		boolean include = true;
		while (i < columns*rows - (Window/2)*columns)
		{
			if (data[i]>MinLevel)
			{
				j = 0;
				k = i-(Window/2)*(columns+1);
				include = true;
				loopC = 0;
				while (k <= i + (Window/2)*(columns+1) && include)
				{
					if (data[k] > data[i])
						include = false;
					if(data[k] > 0)
						j++;
					k++;
					loopC++;
					if (loopC == Window)
					{
						k += (columns-Window);
						loopC = 0;
					}
				}
				if (j < minPosPixels)
					include = false;
				if(data[i-1] >= minNeighbourValue &&
						data[i+1] >= minNeighbourValue &&
						data[i-columns] >=minNeighbourValue &&
						data[i+columns] >= minNeighbourValue &&
						include)
			//	if(include)
				{

						int[] coord = {i%columns,i/columns};					
						Results.add(coord);	
					
				}
			}
			i++;

			if((i % columns )== (columns - Window/2 ))
				i += Window-1;
		}

		return Results;
	}

	/*
	 * identify overlaps.
	 */
	public static ArrayList<int[]> cleanList(ArrayList<int[]> Array, double Dist){
		int[] Start = {0,0};
		int[] Compare = {0,0}; 
		ArrayList<int[]> CheckedArray = new ArrayList<int[]>();
		int[] found = new int[Array.size()];
		for (int i = 0; i < Array.size(); i++){
			found[i] = 0;
		}
		for (int i = 0; i < Array.size(); i++){

			Start = Array.get(i);
			for (int Check = i+1; Check < Array.size(); Check++){				
				Compare = Array.get(Check);
				double sqDist = ((Start[0] - Compare[0])*(Start[0] - Compare[0]) + 
						(Start[1] - Compare[1])*(Start[1] - Compare[1]));				
				if(sqDist < Dist){
					found[i]++;
					found[Check]++;
				}
			}
			if (found[i] == 0){				
				int[] coord = Start;
				CheckedArray.add(coord);
			}
		}
		return CheckedArray; 
	}
}
