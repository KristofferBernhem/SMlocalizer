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

public class TableIO {
	double x, y, frame, sigma_x, sigma_y, precision_x, precision_y, r_square,photons;
	public static ArrayList<Particle> Load(){ // Load data from table.
		ArrayList<Particle> Results = new ArrayList<Particle>();	
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();		
		for (int i = 0; i < tab.size();i++ ){
			double x = tab.getValue("x0", i);
			double y = tab.getValue("y0", i);
			double z = tab.getValue("z0", i);
			int frame = (int) tab.getValue("frame", i);
			int channel = (int) tab.getValue("channel", i);
			double sigma_x = tab.getValue("sigma_x", i);
			double sigma_y = tab.getValue("sigma_y", i);
			double sigma_z = tab.getValue("sigma_z", i);
			double precision_x = tab.getValue("precision_x", i);
			double precision_y = tab.getValue("precision_y", i);
			double precision_z = tab.getValue("precision_z", i);
			double r_square = tab.getValue("r_square", i);
			int photons = (int) tab.getValue("photons", i);
			int include = (int) tab.getValue("include", i);
			Results.add(new Particle(x,y,z,frame,channel,sigma_x,sigma_y,sigma_z,precision_x,precision_y,precision_z,r_square,photons,include));			
		}		
		return Results;

	}

	public static void Store(ArrayList<Particle> Results){		
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		double width = tab.getValue("width", 0);
		double height = tab.getValue("height", 0);
		tab.reset();		

		for (int i = 0; i < Results.size(); i++){

				tab.incrementCounter();

				tab.addValue("x0", Results.get(i).x);
				tab.addValue("y0", Results.get(i).y);
				tab.addValue("z0", Results.get(i).z);
				tab.addValue("frame", Results.get(i).frame);
				tab.addValue("channel", Results.get(i).channel);
				tab.addValue("sigma_x", Results.get(i).sigma_x);
				tab.addValue("sigma_y", Results.get(i).sigma_y);
				tab.addValue("sigma_z", Results.get(i).sigma_z);
				tab.addValue("precision_x", Results.get(i).precision_x);
				tab.addValue("precision_y", Results.get(i).precision_y);
				tab.addValue("precision_z", Results.get(i).precision_z);
				tab.addValue("r_square", Results.get(i).r_square);
				tab.addValue("photons", Results.get(i).photons);
				tab.addValue("include", Results.get(i).include);
				if (i == 0)
				{
					tab.addValue("width", width);
					tab.addValue("height", height);
				}
			}
	tab.show("Results");
	}
}
