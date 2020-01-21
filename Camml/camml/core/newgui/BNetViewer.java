/*
 *  [The "BSD license"]
 *  Copyright (c) 2002-2011, Rodney O'Donnell, Lloyd Allison, Kevin Korb
 *  Copyright (c) 2002-2011, Monash University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *    2. Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *    3. The name of the author may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.*
 *
 *  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 *  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 *  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 *  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
package camml.core.newgui;

import java.awt.Component;
import java.awt.event.MouseEvent;
import java.io.ByteArrayInputStream;


import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.event.MouseInputAdapter;
import javax.swing.JFrame;

import norsys.netica.Environ;
import norsys.netica.Net;
import norsys.netica.NeticaException;
import norsys.netica.Node;
import norsys.netica.NodeList;
import norsys.netica.Streamer;
import norsys.netica.Value;
import norsys.netica.gui.NetPanel;
import norsys.netica.gui.NodePanel;
import norsys.netica.gui.NodePanel_BeliefBars;
import norsys.netica.gui.NodePanel_BeliefBarsRow;



/**Simple Bayesian Network Viewer using Netica Libraries 
 * Displays a Bayesian network parameterized by CaMML, allows user to reposition nodes,
 * and run inference on that network.
 * As BNs generated by CaMML have no layout information, nodes are positioned using a
 * simple grid arrangement.
 * 
 * Note: May not function if Netica library files are not in a location that is
 * 	part of the OS PATH environment variable.
 * 
 * @author Alex Black
 */
public class BNetViewer extends JFrame {
	private static final long serialVersionUID = -977159232927092552L;
	
	private static final String defaultWindowName = "CaMML - Network Viewer";
	protected NetPanel netPanel;
	protected static Environ env = null;				//Netica Environ (required by Netica library)
	protected Net net;									//Reference to the BN
	
	private static final int windowWidth = 1024;		//Default window width
	private static final int windowHeight = 600;		//Default window height
	
	//Graph layout constants:
	private static final int nodeWidth = 180;			//Assumed width of node for layout purposes
	private static final int nodeHeight = 120;			//Assumed height of node for layout purposes
	private static final int nodeSeparation = 50;		//Separation of nodes for layout purposes
	
	/**Constructor - Creates a window (JFrame) to display a Bayesian Network 
	 * Note: Constructor can either take a file path (and load the network from a file) OR
	 * it can take an entire network as a String (i.e. output of NeticaFN.SaveNet.apply(...)
	 * or ExportDBNNetica.makeNeticaFileString(...))
	 * @param network Bayesian Network - the entire network as a String (if isFileName == false)
	 * 		or the path of a network file (if isFileName == true)
	 * @param isFileName True: Load from file (network is file path). False: network specified by string 
	 * @param windowTitle Title of display window
	 * @throws Exception For IO errors, NeticaException, etc
	 */
	public BNetViewer( String network, boolean isFileName, String windowTitle ) throws Exception {

		if( env == null ) env = new Environ(null);
		
		//Network path passed:
		if( isFileName ){
			net = new Net( new Streamer( network ) );
		} else { //Assume actual network is passed as a string...
			//Create from string: (A bit of a hackish work-around, but the String needs to be a type of InputStream...)
			net = new Net( new Streamer(new ByteArrayInputStream( network.getBytes("UTF-8")), "StringReader", env ) );
		}
		
		//Lay out the graph: (By default, all nodes are at (0,0))
		layoutGraph();
			
		net.compile();
		
		// Create a NetPanel for 'net'
		netPanel = new NetPanel(net, NodePanel.NODE_STYLE_AUTO_SELECT);

		// Make all the components listen for mouse clicks
		netPanel.addListenerToAllComponents(new ViewerMouseInputAdapter( this ));

		// Add the panel to the application's content pane
		getContentPane().add(new JScrollPane(netPanel));

		//Set the title for the window
		this.setTitle( windowTitle );
		
		// Close the window (but not program) when the user clicks 'X'
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

		// Set the frame (i.e. window) size and show the frame
		setSize(windowWidth, windowHeight);
		setVisible(true);
	}
	
	/**Alternate constructor. Uses default window name.
	 * See other constructor for details
	 */
	public BNetViewer( String network, boolean isFileName ) throws Exception {
		this( network, isFileName, defaultWindowName );
	}
	
	
	/**Lays the graph out in a basic grid arrangement.
	 * No doubt there are MUCH better methods of doing such a layout, but it's
	 * better than having all the nodes at position (0,0) by default.
	 * 
	 * Grid is set out top left to bottom right according to the order given by
	 *  net.getNodes()
	 */
	public void layoutGraph(){
		//First: Get all Nodes...
		NodeList list;
		try{
			list = net.getNodes();
		} catch( NeticaException e ){
			return;
		}
		
		int numPerRow = (windowWidth - nodeWidth) / (nodeWidth);
		
		for( int i = 0; i < list.size(); i++ ){
			Node n = (Node)list.get(i);													//Current node
			int rowNum = i*(nodeWidth + nodeSeparation) / (windowWidth - nodeWidth);	//Determine row number for this node
			int posY = rowNum * (nodeHeight + nodeSeparation);
			int posX = nodeSeparation + i % numPerRow * ( nodeWidth + nodeSeparation );
			try{
                n.visual().setPosition(posX, posY);
            } catch( NeticaException e ){
                //Do nothing?
            }
        }
    }

    /**Class to deal with mouse events: i.e. click, click+drag on BN nodes
     * Required for inference (click) and repositioning nodes (click+drag).
     * @author Alex Black
     */
    static class ViewerMouseInputAdapter extends MouseInputAdapter {

        int mouseDownStartX = 0;	//Used for moving nodes - click + drag
        int mouseDownStartY = 0;	//Used for moving nodes - click + drag
        Component lastClicked;		//Used for moving nodes - click + drag

        BNetViewer viewer;

        public ViewerMouseInputAdapter(BNetViewer viewer){
            this.viewer = viewer;
        }

        //User clicks mouse:
        public void mouseClicked(MouseEvent me) {
            try {
                // Find out which component got clicked
                Component comp = me.getComponent();

                // If a belief bar was clicked...
                if (comp instanceof NodePanel_BeliefBarsRow) {
                    NodePanel_BeliefBarsRow row = (NodePanel_BeliefBarsRow) comp;

                    // Find the state index of the belief bar
                    int clickedState = row.getState().getIndex();

                    // Get the node that owns this belief bar, and get that node's current finding
                    Value finding    = row.getState().getNode().finding();

                    // If the node finding is what was clicked, clear the finding
                    if (finding.getState() == clickedState) {
                        finding.clear();
                        // Otherwise, set the finding
                    } else {
                        finding.setState(clickedState);
                    }

                    // Compile net and refresh the display
                    viewer.netPanel.getNet().compile();
                    viewer.netPanel.refreshDataDisplayed();
                }
            } catch (NeticaException e) { e.printStackTrace(); }
        }

        //User clicks and holds mouse button down:
        public void mousePressed(MouseEvent me){
            //Store the location (+component) where the click+drag started, for later use:
            mouseDownStartX = me.getX();
            mouseDownStartY = me.getY();
            lastClicked = me.getComponent();
        }

        //User releases mouse after holding down:
        public void mouseReleased(MouseEvent me){
            if( lastClicked == null ) return;

            Component toMove;

            if( lastClicked instanceof NodePanel ){
                toMove = lastClicked;
            } else if( lastClicked instanceof NodePanel_BeliefBarsRow ){
                toMove = ((NodePanel_BeliefBarsRow)lastClicked).getParent().getParent();
            } else if( lastClicked instanceof NodePanel_BeliefBars ){
                toMove = ((NodePanel_BeliefBars)lastClicked).getParent();
            } else if( lastClicked instanceof JLabel ){
                toMove = ((JLabel)lastClicked).getParent();
            } else{
                //Unknown component: Don't know how to deal with it...
                return;
            }

            if( toMove == null ) return;	//Should never happen...

            int newX = toMove.getX() + me.getX() - mouseDownStartX;
            int newY = toMove.getY() + me.getY() - mouseDownStartY;



            //Make sure it cannot be moved outside of window:
            if( newX < 0 ) newX = 0;
            if( newY < 0 ) newY = 0;
            if( newX > viewer.getWidth() - toMove.getWidth()  ) newX = viewer.getWidth() - toMove.getWidth();
            if( newY > viewer.getHeight() - toMove.getHeight() ) newY = viewer.getHeight() - toMove.getHeight();

            toMove.setLocation( newX, newY);

            // Compile net and refresh the display
            try{
                viewer.netPanel.getNet().compile();
                viewer.netPanel.refreshDataDisplayed();
            } catch( NeticaException e){
            }
        }
    }
}