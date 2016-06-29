import java.util.ArrayList;

import ij.plugin.filter.Analyzer;

public class TableIO {
	double x, y, frame, sigma_x, sigma_y, precision_x, precision_y, chi_square,photons;
	public static ArrayList<Particle> Load(){ // Load data from table.
		ArrayList<Particle> Results = new ArrayList<Particle>();	
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();		
		for (int i = 0; i < tab.size();i++ ){
			double x = tab.getValue("x0", i);
			double y = tab.getValue("y0", i);
			double frame = tab.getValue("frame", i);
			double sigma_x = tab.getValue("sigma_x", i);
			double sigma_y = tab.getValue("sigma_y", i);
			double precision_x = tab.getValue("precision_x", i);
			double precision_y = tab.getValue("precision_y", i);
			double chi_square = tab.getValue("chi_square", i);
			double photons = tab.getValue("photons", i);
			Results.add(new Particle(x,y,frame,sigma_x,sigma_y,precision_x,precision_y,chi_square,photons));			
		}		
		return Results;
		
	}
	
	public static void Store(ArrayList<Particle> Results){		
		ij.measure.ResultsTable tab = Analyzer.getResultsTable();
		tab.reset();		
		for (int i = 0; i < Results.size(); i++){
			tab.incrementCounter();
			tab.addValue("x0", Results.get(i).x);
			tab.addValue("y0", Results.get(i).y);
			tab.addValue("frame", Results.get(i).frame);
			tab.addValue("sigma_x", Results.get(i).sigma_x);
			tab.addValue("sigma_y", Results.get(i).sigma_y);
			tab.addValue("precision_x", Results.get(i).precision_x);
			tab.addValue("precision_y", Results.get(i).precision_y);
			tab.addValue("chi_square", Results.get(i).chi_square);
			tab.addValue("photons", Results.get(i).photons);
		}
		tab.show("Results");
	}
}
