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
import java.util.ArrayList;

public class renderImage {
	public static void run( int[] DesiredPixelSize ){		
		try{
			ArrayList<Particle> correctedResults	= TableIO.Load(); // load dataset from results table.
			int Width 								= 0; // will contain the largest width value of the dataset.
			int Height 								= 0; // will contain the largest height value of the dataset.
			for (int i = 0; i < correctedResults.size();i++){
				if (correctedResults.get(i).include == 1){ 

					if (Math.round(correctedResults.get(i).x) > Width){ // if the new value is larer.
						Width = (int) Math.round(correctedResults.get(i).x); // update max value.
					}
					if (Math.round(correctedResults.get(i).y) > Height){ // if the new value is larger.
						Height = (int) Math.round(correctedResults.get(i).y); // update max value.
					}
				}
			}		
			Width 	= Width + 10; //inputPixelSize[0]; 	// pad with one pixel.
			Height 	= Height + 10; //inputPixelSize[0]; 	// pad with one pixel.
			generateImage.create("RenderedResults",correctedResults, Width, Height, DesiredPixelSize);		
		}
		catch (Exception e){
			ij.IJ.log("No results table found.");	
		}		
	}	
}
