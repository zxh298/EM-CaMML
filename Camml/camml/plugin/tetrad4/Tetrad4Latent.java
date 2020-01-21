package camml.plugin.tetrad4;

import java.util.ArrayList;
import java.util.List;

import camml.core.search.TOM;

//import org.apache.log4j.Logger;

import camml.plugin.tetrad4.*;
import camml.plugin.weka.Converter;
import cdms.core.Type;
import cdms.core.Value;
import edu.cmu.tetrad.data.Knowledge;
import edu.cmu.tetrad.data.RectangularDataSet;
import edu.cmu.tetrad.graph.Dag;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphNode;
import edu.cmu.tetrad.graph.Node;
import edu.cmu.tetrad.search.IndTestChiSquare;
import edu.cmu.tetrad.search.IndTestGSquare;
import edu.cmu.tetrad.search.IndependenceTest;

public class Tetrad4Latent extends Tetrad4{
	
	public static TOM getLatentTOM(int[][] latent_matrix, Value.Vector data)
	{
		String[] labels = ((Type.Structured)((Type.Vector)data.t).elt).labels;
		
		List<Node> nodelist = new ArrayList<Node>();
		for(int i=0; i<labels.length; i++)
		{
			if(labels[i] != null)
			{ 
				Node node = new GraphNode(labels[i]);
				nodelist.add(node);
			}
			else
			{
				String label = "var(" + i +")";
				Node node = new GraphNode(label);
				nodelist.add(node);	
			}
		}
			
		Dag dag = new Dag(nodelist); 
		 
		for(int n=0; n<latent_matrix.length; n++)
		{
			for(int m=0; m<latent_matrix[n].length; m++)
			{
				if(latent_matrix[n][m] == 1)
					dag.addDirectedEdge(nodelist.get(n), nodelist.get(m));
			}
		}
	     
	     TOM tom = Tetrad4.dagToTOM(data, dag);
	     
	     return tom;
	}
	
}
