import com.typesafe.config.ConfigFactory

import java.io._
import java.nio.file.{Paths,Files}

import scala.io.Source


object ReRun {


    val configFile = "ealReRun.conf"
    val config = ConfigFactory.load(configFile)

    val dfIn = config.getString("df-in")
    val nwIn = config.getString("nw-in")
    val dfZeros = config.getString("df-zeros")
    val nwZeros = config.getString("nw-zeros")
    val nytZeros = config.getString("nyt-zeros")



    def main(args: Array[String]) {

       val dfZero = {
         if(!Files.exists(Paths.get(dfZeros))){
           System.out.println(s"Input file $dfZeros doesn't exist! " + s"Exiting...")
           sys.exit(1)
         }
         //Read file, line by line
         Source.fromFile(dfZeros).getLines().map(line => {
           line.trim
         })
       }.toList.toSet
       //System.out.println("Num df IDs: " + dfZero.size)

       val nwZero = {
         if(!Files.exists(Paths.get(nwZeros))){
           System.out.println(s"Input file $nwZeros doesn't exist! " + s"Exiting...")
           sys.exit(1)
         }
         //Read file, line by line
         Source.fromFile(nwZeros).getLines().map(line => {
           line.trim
         })
       }.toList.toSet
       //System.out.println("Num df IDs: " + nwZero.size)

       val nytZero = {
         if(!Files.exists(Paths.get(nytZeros))){
           System.out.println(s"Input file $nytZeros doesn't exist! " + s"Exiting...")
           sys.exit(1)
         }
         //Read file, line by line
         Source.fromFile(nytZeros).getLines().map(line => {
           line.trim
         })
       }.toList.toSet
       //System.out.println("Num df IDs: " + nytZero.size)


       //if(!Files.exists(Paths.get(dfIn))){
       //  System.out.println(s"Input file $dfIn doesn't exist! " + s"Exiting...")
       //   sys.exit(1)
       //}

       //var fileName = dfIn
       var fileName = nwIn

       var docid = "newDoc"
       //Read file, line by line
       var lineCount = 0
       var relationsCount = 0
       var posRelationsCount = 0
       val lines = Source.fromFile(fileName).getLines().toList

       for(line <- lines){
           lineCount += 1

           if(line.startsWith("doc #")){
              docid = line.split("docId: ")(1)              
           }
           if(line.startsWith("RELATIONS:")){
              relationsCount += 1
              var relations = line.split("RELATIONS: ")(1).toInt            
              if(relations > 0 && nytZero.contains(docid+".xml")){
              //if(relations > 0 && nwZero.contains(docid+".xml")){
              //if(relations > 0 && dfZero.contains(docid+".xml")){
                  posRelationsCount += 1
                  //System.out.println(docid + " " + relations.toInt)
                  System.out.println(docid + ".xml")
              }
           }
       }
       //System.out.println("df lines: " + lines.size)
       //System.out.println("relationsCount: " + relationsCount)
       //System.out.println("posRelationsCount: " + posRelationsCount)


    } //main

}
