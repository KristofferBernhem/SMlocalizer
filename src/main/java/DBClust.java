import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.clustering.DoublePoint;

public class DBClust {
	
	public static void main(String[] args){	
		// test function
		int nFrames = 30;
		int width	= 12800;
		int height 	= 12800;
		int perFrame= 1000;
		int totalParticles = 1000;
		ArrayList<Particle> testResults = TestData.generate(nFrames, width, height, perFrame,totalParticles);
		
		
		
		double eps = 10; // Search radius around each point.
		int minPts = 3; // minimum connectivity.
		List<Cluster<DoublePoint>> outp = Ident(eps,minPts,testResults);
	    for(Cluster<DoublePoint> c: outp){ // how to get access to all clusters.
	        System.out.println((c.getPoints().get(0)) +" "+  c.getPoints().size());
	        
	    }   
		System.out.println(outp.size()); // Total number of clusters found.			
	}

	public static List<Cluster<DoublePoint>> Ident(double eps, int minPts, ArrayList<Particle> InpParticle){
		List<DoublePoint> points = new ArrayList<DoublePoint>();

		for (int i = 0; i < InpParticle.size(); i++){
			double[] p = new double[2];
			p[0] = InpParticle.get(i).x;
			p[1] = InpParticle.get(i).y;
			points.add(new DoublePoint(p));
		}

		DBSCANClusterer<DoublePoint> DB = new DBSCANClusterer<DoublePoint>(eps, minPts);

		return DB.cluster(points); 
	}


}
