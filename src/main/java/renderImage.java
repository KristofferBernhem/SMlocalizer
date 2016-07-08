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
	public static void run(int inputPixelSize, int DesiredPixelSize ){		
		try{
			ArrayList<Particle> correctedResults = TableIO.Load();	
			int Width 				= 0;
			int Height 				= 0;
	//		int Pad					= inputPixelSize;

			//int PixelRatio = Math.round(inputPixelSize/DesiredPixelSize);
			for (int i = 0; i < correctedResults.size();i++){
				if (correctedResults.get(i).include == 1){ 

					if (Math.round(correctedResults.get(i).x) > Width){
						Width = (int) Math.round(correctedResults.get(i).x);
					}
					if (Math.round(correctedResults.get(i).y) > Height){
						Height = (int) Math.round(correctedResults.get(i).y);
					}
				}
			}		
			Width = Width + inputPixelSize;
			Height = Height + inputPixelSize;
			generateImage.create("RenderedResults",correctedResults, Width, Height, DesiredPixelSize);		
		}
		catch (Exception e){
		}
	}
}
