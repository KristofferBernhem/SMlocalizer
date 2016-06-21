import java.util.ArrayList;

/*
 * 
 */
public class LocalMaxima {
	public static ArrayList<int[]> FindMaxima(double[][] Array, double SN, double Noise){
		ArrayList<int[]> Results = new ArrayList<int[]>();
		//int[] XY = {5,3}; //Example of how results are organized.		
		//Results.add(XY);
		double MinLevel = Noise*SN; // Do this calc only once.
		//ArrayList<int[]> Candidates = new ArrayList<int[]>();
		
		for (int i = 2; i < Array.length-2;i++){ // Look through all columns except outer 2.
			for (int j = 2; j < Array[0].length-2; j++){ // Look through all rows except outer 2.
				if (Array[i][j] < Noise){ // If values are below noise level, turn it off.
					Array[i][j] = 0;
				}
				else if(Array[i][j] > MinLevel &&
						Array[i-1][j-1] > MinLevel && 
						Array[i][j-1] > MinLevel &&
						Array[i+1][j-1] > MinLevel &&
						Array[i-1][j] > MinLevel &&
						Array[i+1][j] > MinLevel &&
						Array[i-1][j+1] > MinLevel &&
						Array[i+1][j+1] > MinLevel &&
						Array[i-1][j-1] < Array[i][j] && 
						Array[i][j-1] < Array[i][j] &&
						Array[i+1][j-1] < Array[i][j] &&
						Array[i-1][j] < Array[i][j] &&
						Array[i+1][j] < Array[i][j] &&
						Array[i-1][j+1] < Array[i][j] &&
						Array[i+1][j+1] < Array[i][j]){
					int[] coord = {i,j};					
					Results.add(coord);					
				}

			}
		}
		//System.out.println(Results.size());
		Results = Neighbours(Results,5);
		
		// Clean out Results based on strongest 

		return Results;
	}

	/*
	 * identify overlaps.
	 */
	public static ArrayList<int[]> Neighbours(ArrayList<int[]> Array, int Dist){
		int[] Start = {0,0};
		int[] Compare = {0,0}; 
		ArrayList<int[]> Ignore = new ArrayList<int[]>();
		
		for (int i = 0; i < Array.size(); i++){
			Start = Array.get(i);
			for (int Check = i+1; Check < Array.size(); Check++){				
				Compare = Array.get(Check);
				if (Math.abs(Start[0] - Compare[0]) < Dist &&
						Math.abs(Start[1] - Compare[1]) < Dist){
					int[] entry = {Check};
					Ignore.add(entry);
					
				}
			}
			if( !Ignore.isEmpty()){
				for (int j = Ignore.size()-1; j >= 0; j--){
					int[] entry = Ignore.get(j);
					Array.remove(entry[0]);
				}
				Array.remove(i);
				Ignore.clear();
			}
			
			
		}
//		return Ignore;
		return Array; 
	}
}