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

//
// Module for Tetrad4 interface.
//

// File: RodoCamml.java
// Author: rodo@csse.monash.edu.au

package camml.plugin.tetrad4;

import cdms.core.*;

/**
 * Module to interface with Tetrad IV functions.  <br>
 *  Note: This plugin requires a copy of Tetrad4.jar in the classpath.
 *   
 * @author Rodney O'Donnell <rodo@dgs.monash.edu.au>
 * @version $Revision$ $Date$
 * $Source$
 */
public class Tetrad4Module extends Module {
    public static java.net.URL helpURL = Module
        .createStandardURL(Tetrad4Module.class);

    public String getModuleName() {
        return "Tetrad4Module";
    }

    public java.net.URL getHelp() {
        return helpURL;
    }

    public void install(Value params) throws Exception {
        add("GES", TetradLearner.ges.getFunctionStruct(), "Tetrad GES functions. Member of SEC chosen at random.");
        add("FCI", TetradLearner.fci.getFunctionStruct(), "Tetrad SEC functions. Member of SEC chosen at random.");
        add("PC", TetradLearner.pcRepair.getFunctionStruct(), 
            "Tetrad PC functions. Member of SEC chosen at random. Broken SECs repaired.");
        if ( params instanceof Value.Vector) {
            Value.Vector pVec = (Value.Vector)params;
            for ( int i = 0; i < pVec.length(); i++) {
                Value.Structured elt = (Value.Structured)pVec.elt(i);
                String s = ((Value.Str)elt.cmpnt(0)).getString();
                if (s.equals("stringParams") ) { 
                    TetradLearner.stringParams = (elt.cmpnt(1) == Value.TRUE);
                    System.out.println("stringParams = " + TetradLearner.stringParams);
                }
                else if (s.equals("useVariableNames")) {
                    TetradLearner.useVariableNames = (elt.cmpnt(1) == Value.TRUE);
                    System.out.println("stringParams = " + TetradLearner.useVariableNames);
                    
                }
            }
        }
    }


}
