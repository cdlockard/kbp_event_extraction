import com.typesafe.config.ConfigFactory

import java.io._
import java.nio.file.{Paths,Files}

import scala.io.Source


object CombineEALResults {


    val configFile = "ealCombine.conf"
    val config = ConfigFactory.load(configFile)

    val sys1Dir = config.getString("sys1-dir")
    val sys2Dir = config.getString("sys2-dir")
    val sys3Dir = config.getString("sys3-dir")

    val outputDir = config.getString("output-dir")
    val docIDList = config.getString("docID-list")    

    //val matchOnly = config.getString("match-only")

    case class Response(id:String, docid:String, event:String, role:String, cas:String, casBegin:Int,
      casEnd:Int, predJustBegin:Int, predJustEnd:Int, baseFillerBegin:Int, baseFillerEnd:Int, 
      additionalJustification:String, realis:String, confidence:String)

    case class ResponseKey(event:String, role:String, baseFillerOffsets:String)

    case class Linking(id:String, args:String)


    def main(args: Array[String]) {

      // ------------------------
      // Print config args
      // ------------------------
      System.out.println("Config File: " + configFile)
      System.out.println("Input: ")
      System.out.println("sys1Dir: " + sys1Dir)
      System.out.println("sys2Dir: " + sys2Dir)
      System.out.println("sys3Dir: " + sys3Dir)
      System.out.println("Output: ")
      System.out.println("outputDir: " + outputDir)
      System.out.println("DocIDs: ")
      System.out.println("docIDList: " + docIDList)
      //System.out.println("matchOnly: " + matchOnly)
      System.out.println("\nReading input data..." + "\n")


      // ---------------------------------------
      // Read list of docIDs in Eval Corpus
      // ---------------------------------------
      val docIDs = {
        if(!Files.exists(Paths.get(docIDList))){
          System.out.println(s"Input file $docIDList doesn't exist! " + s"Exiting...")
          sys.exit(1)
        }
        //Read file, line by line
        Source.fromFile(docIDList).getLines().map(line => {
          line.trim
         })
      }.toList
      System.out.println("Num docIDs: " + docIDs.size)

      docIDs.foreach(d =>{

        System.out.println("Processing doc: " + d)
        val docID = d

        // -----------------------------------
        // read sys1 output - args
        // -----------------------------------
        val sys1Args = getSysArgs(sys1Dir, docID)
        System.out.println("sys1: " + sys1Args.size)

        // ------------------------------------
        // read sys1 output - linking
        // ------------------------------------
        val sys1Linking = getSysLinking(sys1Dir, docID)
        System.out.println("sys1Linking: " + sys1Linking.size)

        var sys1LinkArgsSet = getSysLinkArgsSet(sys1Linking, sys1Args)
        System.out.println("sys1LinkArgsSet: " + sys1LinkArgsSet.size)

        //for(argset <- sys1LinkArgsSet){
        //  for(response <- argset) {
        //    System.out.println("response: " + responseToString(response))            
        //  }
        //}

        // ------------------------------------
        // read sys2 output - args
        // ------------------------------------

        val sys2Args = getSysArgs(sys2Dir, docID)
        System.out.println("sys2: " + sys2Args.size)

        // -------------------------------------
        // read sys2 output - linking
        // -------------------------------------

        val sys2Linking = getSysLinking(sys2Dir, docID)
        System.out.println("sys2Linking: " + sys2Linking.size)

        var sys2LinkArgsSet = getSysLinkArgsSet(sys2Linking, sys2Args)
        System.out.println("sys2LinkArgsSet: " + sys2LinkArgsSet.size)

        // -------------------------------------
        // read sys3 output - args
        // -------------------------------------

        val sys3Args = getSysArgs(sys3Dir, docID)
        System.out.println("sys3: " + sys3Args.size)

        // --------------------------------------
        // read sys3 output - linking
        // --------------------------------------

        val sys3Linking = getSysLinking(sys3Dir, docID)
        System.out.println("sys3Linking: " + sys3Linking.size)

        var sys3LinkArgsSet = getSysLinkArgsSet(sys3Linking, sys3Args)
        System.out.println("sys3LinkArgsSet: " + sys3LinkArgsSet.size)

       // ----------------------------------------
       // sys Combined 
       // ----------------------------------------

       val matchOnly = true

       //val sysCombinedLinkArgsSet = combine2Systems(combine2Systems(sys1LinkArgsSet, sys2LinkArgsSet, matchOnly), sys3LinkArgsSet, matchOnly)
       val sysCombinedLinkArgsSet = combine2Systems(sys1LinkArgsSet, sys2LinkArgsSet, matchOnly)

       //val sysCombinedLinkArgsSet = combine2Systems(combine2Systems(sys1LinkArgsSet, sys2LinkArgsSet), sys3LinkArgsSet)
       
       //val sysCombinedLinkArgsSet = combine2Systems(sys1LinkArgsSet, sys2LinkArgsSet, matchOnly)       

       //This would create a grand sum of the three systems
       //val sysCombinedLinkArgsSet = sys1LinkArgsSet ++ sys2LinkArgsSet ++ sys3LinkArgsSet

       System.out.println("-----sysCombined: " + sysCombinedLinkArgsSet.size)


       //System.out.println("Combined ----------------------------------------")
       //System.out.println("sys1LinkArgsSet:" + sys1LinkArgsSet.size)
       //System.out.println("sys2LinkArgsSet:" + sys2LinkArgsSet.size)
       //System.out.println("sys3LinkArgsSet:" + sys3LinkArgsSet.size)
       //System.out.println("sysCombinedLinkArgsSet:" + sysCombinedLinkArgsSet.size)
       //System.out.println("Combined ----------------------------------------")

       printSystemOutput(docID, sysCombinedLinkArgsSet)


      }) //foreach docID

      //System.out.println(lowerString("supersonics") + "!!")

    } //main


    //def responseToString(response:Response): String = {
    // 
    // val responseString = response.docid + "\t" + response.event + "\t" + response.role + "\t" + 
    //    response.cas + "\t" + response.casOffsets + "\t" + response.predicateJustification + "\t" +
    //    response.baseFillerOffsets + "\t" + response.additionalJustification + "\t" + 
    //    response.realis + "\t" + response.confidence      
    //
    //  return responseString
    //}

    def responseToString(response:Response): String = {
     
      val responseString = response.docid + "\t" + response.event + "\t" + response.role + "\t" + 
        response.cas + "\t" + response.casBegin + "-" + response.casEnd + "\t" + 
        response.predJustBegin + "-" + response.predJustEnd + "\t" +
        response.baseFillerBegin + "-" + response.baseFillerEnd + "\t" + 
        response.additionalJustification + "\t" + response.realis + "\t" + response.confidence      
    
      return responseString
    }


    def getSysArgs(sysDir:String, docID:String):List[Response] = {
    
      val sysArgs = {

          val fileName = sysDir+"arguments/"+docID
          if(!Files.exists(Paths.get(fileName))){
            System.out.println(s"Input file $fileName doesn't exist! " + s"Exiting...")
            sys.exit(1)
          }

          //Read file, line by line
          Source.fromFile(fileName).getLines().map(line => {

            val fields = line.trim.split("\t")
            if(fields.size >= 11){
              val casOffsets = fields(5).split("-")
              var casBegin = 0
              var casEnd = 0
              if(casOffsets.size >=2){
                casBegin =  casOffsets(0).toInt
                casEnd = casOffsets(1).toInt
              }
              val pjOffsets = fields(6).split("-")
              var predJustBegin = 0
              var predJustEnd = 0
              if(pjOffsets.size >=2){
                predJustBegin =  pjOffsets(0).toInt
                predJustEnd = pjOffsets(1).toInt
              }
              val bfOffsets = fields(7).split("-")
              var baseFillerBegin = 0
              var baseFillerEnd = 0
              if(bfOffsets.size >=2){
                baseFillerBegin =  bfOffsets(0).toInt
                baseFillerEnd = bfOffsets(1).toInt
              }

              Response(fields(0),fields(1),fields(2),fields(3),fields(4),
                casBegin,casEnd,predJustBegin,predJustEnd,baseFillerBegin,baseFillerEnd,
                fields(8),fields(9),fields(10))
            }
            else{
              System.out.println("****getSysArgs: badLine")
              Response("badLine","fields(1)","fields(2)","fields(3)","fields(4)",-1,-1,
                -1,-1,-1,-1,"fields(8)","fields(9)","fields(10)")
            }

          }).toList.filter(r => r.id != "badLine")
        }

        return sysArgs
      
    }

    def getSysLinking(sysDir:String, docID:String):List[Linking] = {

      val sysLinking = {

          val fileName = sysDir+"linking/"+docID
          if(!Files.exists(Paths.get(fileName))){
            System.out.println(s"Input file $fileName doesn't exist! " + s"Exiting...")
            sys.exit(1)
          }

          Source.fromFile(fileName).getLines().map(line => {

            val linksLine = line.trim.split("\t")

            if(linksLine.size > 1){
              Linking(linksLine(0), linksLine(1))
            }
            else{
              System.out.println("****sysLinking: badLine")
              Linking("badLine", "linksLine(1)")
            }

          }).toList.filter(l => l.id != "badLine")

        }

      return sysLinking

    }


    def getSysLinkArgsSet(sysLinking:List[Linking], sysArgs:List[Response]):
      scala.collection.mutable.Set[scala.collection.mutable.Set[Response]] = {

        var sysLinkArgsSet = scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]()
        for(link <- sysLinking){
          val linkArgs = link.args.split("\\s")
          var sysResponseSet = scala.collection.mutable.Set[Response]()
          linkArgs.foreach(a => {
            val response = sysArgs.filter(k => k.id==a)(0)
            sysResponseSet.add(response)
          })
            sysLinkArgsSet.add(sysResponseSet)
            System.out.println("sysResponseSet: " + sysResponseSet.size)
        }

        return sysLinkArgsSet  
    }

    def combine2Systems(sys1:scala.collection.mutable.Set[scala.collection.mutable.Set[Response]],
      sys2:scala.collection.mutable.Set[scala.collection.mutable.Set[Response]],
      matchOnly:Boolean):scala.collection.mutable.Set[scala.collection.mutable.Set[Response]] = {

        var sysCombined = scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]()

        (sys1.size, sys2.size) match {

          case p: Pair[Int,Int] if (p._1 >0 && p._2 >0) => {
            System.out.println("!!!Both systems have values!!")
            if(!matchOnly) {sysCombined = compare2Systems(sys1,sys2)}
            else{ sysCombined = compare2SystemsMatch(sys1,sys2)}
          }
          case p: Pair[Int,Int] if (p._1 >0 && p._2 == 0) => {
            System.out.println("!!!Only sys1 has values!!")
            if(!matchOnly) {sysCombined = sys1}
          }
          case p: Pair[Int,Int] if (p._1 == 0 && p._2 >0) => {
            System.out.println("!!!Only sys2 has values!!")
            if(!matchOnly) {sysCombined = sys2}
          }

          case _ => System.out.println("Neither system has values!")        

        }

        return sysCombined

    }

    def compare2Systems(sys1:scala.collection.mutable.Set[scala.collection.mutable.Set[Response]],
      sys2:scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]):
      scala.collection.mutable.Set[scala.collection.mutable.Set[Response]] = {

        var sysCombined = scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]()
        var sys2MatchingEvents = scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]()    
        
        var linkEvent = ""
        var predJustBegin = 0 
        var predJustEnd = 0

        for(link <- sys1){
           for(arg <- link){
              linkEvent = arg.event
              predJustBegin = arg.predJustBegin
              predJustEnd = arg.predJustEnd
           }
           sys2MatchingEvents = sys2.filter(s => matchingEvent(s,linkEvent,predJustBegin,predJustEnd))
           System.out.println("####CB size: " + linkEvent + " " + sys2MatchingEvents.size) 
           if(sys2MatchingEvents.size == 0) {                
             sysCombined += link
           }
           //check args
           else {
                var argSet = scala.collection.mutable.Set[Response]()
                sys2MatchingEvents.foreach(l => {    
                  l.foreach(a => {
                    var newArg = true
                    for(arg <- link){
                      //add sys1 arg
                      argSet += arg
                      if(overlapOffsets(arg.baseFillerBegin,arg.baseFillerEnd,a.baseFillerBegin,a.baseFillerEnd)){
                        newArg = false
                      }
                    }
                    //add sys2 arg, if not in sys1 arg set
                    if(newArg){ argSet += a}
                  })
                  sys2 -= l
                })
                //remove matching from sys2
                sys2MatchingEvents.foreach(s => {sys2 -= s})
                sysCombined += argSet                           
           }
           
        }
        //add the non-matching events from sys2 to the combined
        sys2.foreach(s => {sysCombined += s})
        

        return sysCombined

    }

    def compare2SystemsMatch(sys1:scala.collection.mutable.Set[scala.collection.mutable.Set[Response]],
      sys2:scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]):
      scala.collection.mutable.Set[scala.collection.mutable.Set[Response]] = {

        var sysCombined = scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]()
        var sys2MatchingEvents = scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]()    
        
        var linkEvent = ""
        var predJustBegin = 0 
        var predJustEnd = 0

        for(link <- sys1){
           for(arg <- link){
              linkEvent = arg.event
              predJustBegin = arg.predJustBegin
              predJustEnd = arg.predJustEnd
           }
           sys2MatchingEvents = sys2.filter(s => matchingEvent(s,linkEvent,predJustBegin,predJustEnd))
           System.out.println("####CB size: " + linkEvent + " " + sys2MatchingEvents.size) 
           if(sys2MatchingEvents.size == 0) {                
             //sysCombined += link
           }
           //check args
           else {
                var argSet = scala.collection.mutable.Set[Response]()
                sys2MatchingEvents.foreach(l => {    
                  l.foreach(a => {
                    //var newArg = true
                    for(arg <- link){
                      //add sys1 arg, if sys2 also has it
                      if(overlapOffsets(arg.baseFillerBegin,arg.baseFillerEnd,a.baseFillerBegin,a.baseFillerEnd)){
                        argSet += arg
                        //newArg = false
                      }
                    }
                    //add sys2 arg, if not in sys1 arg set
                    //if(newArg){ argSet += a}
                  })
                  //sys2 -= l
                })
                //remove matching from sys2
                //sys2MatchingEvents.foreach(s => {sys2 -= s})
                if(argSet.size > 0) { sysCombined += argSet }                          
           }
           
        }
        //add the non-matching events from sys2 to the combined
        //sys2.foreach(s => {sysCombined += s})
        

        return sysCombined

    }


    def matchingEvent(argSet:scala.collection.mutable.Set[Response],linkEvent:String,
      predJustBegin:Int, predJustEnd:Int):Boolean = {
      
      var matches = false
    
      for(response <- argSet){
        if(response.event == linkEvent && overlapOffsets(predJustBegin,predJustEnd,response.predJustBegin,response.predJustEnd)) {
          matches = true
        }
      }  

      return matches

    }

    def overlapOffsets(predJustBegin1:Int,predJustEnd1:Int,predJustBegin2:Int,predJustEnd2:Int):Boolean = {
    
      var overlap = false

      if( (predJustBegin2 >= predJustBegin1) && (predJustBegin2 <= predJustEnd1) ) overlap = true

      if( (predJustEnd2 >= predJustBegin1) && (predJustEnd2 <= predJustEnd1) ) overlap = true
   
      return overlap

    }

    def matchingEvent(argSet:scala.collection.mutable.Set[Response],linkEvent:String):Boolean = {
      
      var matches = false
    
      for(response <- argSet){
        if(response.event == linkEvent) {
          matches = true
        }
      }  

      return matches

    }

    def printSystemOutput(docID:String, 
      sysCombinedLinkArgsSet:scala.collection.mutable.Set[scala.collection.mutable.Set[Response]]) = {

        // ----------------------------------------------
        // Create outstream for Args and Linking
        // ----------------------------------------------
        val outStreamArgs = new PrintStream(outputDir+"arguments/"+docID)                 
        val outStreamLinking = new PrintStream(outputDir+"linking/"+docID)                 
 
        // -----------------
        // For each link,
        // -----------------
        var linkCount = 0          
        var argCount = 0

        sysCombinedLinkArgsSet.foreach(l => {                    
        
          linkCount += 1
          var argString = ""

          // ------------------------
          // Write out the args        
          // ------------------------
          l.foreach(a => {

            argCount += 1        
            outStreamArgs.println(argCount + "\t" + responseToString(a))
            argString = argString + argCount + " " 
        
          })
        
          outStreamLinking.println(linkCount + "\t" + argString)          
        
        })

        outStreamArgs.close()
        outStreamLinking.close()

    }


} //object 
