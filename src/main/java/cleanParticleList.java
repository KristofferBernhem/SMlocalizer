import java.util.ArrayList;

public class cleanParticleList {

	public static void run(){
		// Pull in user data.
		double[] lb = {
				0,								// Allowed lower range of sigma_x in nm, user input.
				0,								// Allowed lower range of sigma_y in nm, user input.
				0,								// Allowed lower range of sigma_z in nm, user input.
				0,								// Allowed lower range of precision_x in nm, user input.
				0,								// Allowed lower range of precision_y in nm, user input.
				0,								// Allowed lower range of precision_z in nm, user input.
				0,								// Allowed lower range of chi_square, user input.
				100								// Allowed lower range of photon count, user input.
		};  				
		double[] ub = {
				300,						// Allowed upper range of sigma_x in nm, user input.
				300,						// Allowed upper range of sigma_y in nm, user input.
				300,						// Allowed upper range of sigma_z in nm, user input.
				300,						// Allowed upper range of precision_x in nm, user input.
				300,						// Allowed upper range of precision_y in nm, user input.
				300,						// Allowed upper range of precision_z in nm, user input.
				1.0,						// Allowed upper range of chi_square, user input.
				500000000					// Allowed upper range of photon count, user input.
		};


		ArrayList<Particle> Results = TableIO.Load();

		for (int i = 0; i < Results.size(); i++){
			if(Results.get(i).sigma_x <= ub[0] && // Check that all values are within acceptable ranges.
					Results.get(i).sigma_x >= lb[0] &&
					Results.get(i).sigma_y <= ub[1] &&
					Results.get(i).sigma_y >= lb[1] &&
					Results.get(i).sigma_z <= ub[2] &&
					Results.get(i).sigma_z >= lb[2] &&	
					Results.get(i).precision_x <= ub[3] &&
					Results.get(i).precision_x >= lb[3] &&
					Results.get(i).precision_y <= ub[4] &&
					Results.get(i).precision_y >= lb[4] &&	
					Results.get(i).precision_z <= ub[5] &&
					Results.get(i).precision_z >= lb[5] &&	
					Results.get(i).chi_square <= ub[6] &&
					Results.get(i).chi_square >= lb[6] &&
					Results.get(i).photons <= ub[7] &&
					Results.get(i).photons >= lb[7]){
				Results.get(i).include = 1; // Include particle.
			}else {
				Results.get(i).include = 0; // Exclude particle
			}

		}

	}
	public static void delete(){ // Remove entries outside of parameter range.
		run(); // Clean out data.
		ArrayList<Particle> Results = TableIO.Load();
		ArrayList<Particle> cleanedList = new ArrayList<Particle>();
		for (int i = 0; i < Results.size(); i++){
			if (Results.get(i).include == 1){
				cleanedList.add(Results.get(i));
			}
		}
		TableIO.Store(cleanedList);
	}
}
