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

import ij.plugin.filter.Analyzer;

public class RenderIm {
	public static void run(boolean[] renderCh, int[] DesiredPixelSize, boolean gSmoothing){		
	//	try{
			ArrayList<Particle> correctedResults	= TableIO.Load(); // load dataset from results table.
			ij.measure.ResultsTable tab = Analyzer.getResultsTable();
			int Width = (int) tab.getValue("width", 0);
			int Height = (int) tab.getValue("height", 0);
			generateImage.create("RenderedResults",renderCh,correctedResults, Width, Height, DesiredPixelSize, gSmoothing);		
		//}
		//catch (Exception e){
		//	ij.IJ.log("No results table found.");	
	//	}		
	}	
}
