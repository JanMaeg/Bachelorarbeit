Deutscher Romankorpus (DROC)
-----

This repository contains a manually annotated corpus for german literary novels.
DROC contains 90 fragments of novels with an average length of about 200 sentences and a total length of 390.000 tokens.

DROC contains manually labeled annotations for:

1. Character References that refer to (usually human) entities appearing in the novel (about 50.000)
2. Coreferences between those references
3. Direct Speech annotations (about 2000)
4. Speaker and Addressees for each direct speech 

Folder Structure
--

src/main/java.... (contains source code to replicate Inter annotator agreement)
src/main/resources (contains the UIMA typesystem, and metadata for our corpus)
DROC-xmi (contains the corpus as .xmi files, described below)
DROC-TEI (contains the corpus as tei-xml files, the file DROC-RELEASE is the full corpus in a single document)
DOCS-RATER1(documents of annotator1 used to derive the IAA)
DOCS-RATER2(documents of annotator2 used to derive the IAA)



Accessing the content from the different formats
------

You can find the corpus in 2 different formats:

1) Apache UIMA ".xmi"
-

If you want to use this format you could either make use of the Apache UIMA api (and combine with the typesystem src/main/resources/CorefTypeSystem.xml)  or you could try to parse it yourself as a standard xml document.

An .xmi file is built as follows:

The text of the .xmi can be found in the (in our case unique for each document) xml element <cas:Sofa> stored as the attribute sofaString, a small example
```html
<cas:NULL xmi:id="0"/><cas:Sofa xmi:id="1" sofaNum="1" sofaID="_InitialView" mimeType="text" sofaString="TEXT CAN BE FOUND HERE...."          
```

Additionally you want to read the annotations from the  ".xmi" document. In order to access the annotations we made you have to find the following tags in the .xmi document:

For Named Entities you have to look for the <type:NamedEntity...\> tag

 A small example:
```html
    <type:NamedEntity xmi:id="514" sofa="1" begin="1908" end="1915" Name="Kyrilla" ID="8" CoreNamedEntity="true" Numerus="si" CoreRange="1908:1915" NEType="Core"/>
```
This represents a single character reference, next you need to interpret its features as follows:
1. xmi:id, the id of this annotation
2. sofa, this is only relevant if you work within UIMA
3. begin, the character offset in the text where the annotation starts
4. end, the character offset in the text where the annotation ends
5. Name, was relevant for annotating
6. ID, the id of the entity this reference refers to
7. CoreNamedEntitiy, a string showing whether it is a core entity or not, irrelevant if you access this information from the attribute "NEType"
8. Numerus, a string showing the number of the reference
9. CoreRange, a range (as character offsets) displaying the text snippet that is a true proper name
10. NEType, the type of the character reference

Accessing the direct speech alongside speaker and "spokenTo" is as follows:

Find the tag <type:DirectSpeech>, a small example:
```html
<type:DirectSpeech xmi:id="3076" sofa="1" begin="2940" end="2970" Speaker="738" SpokenTo="710" Category="directspeech"/>
```

Then parse its features:
1. xmi:id, the id of this annotation
2. sofa, not important
3. begin, the character offset in the text where the annotation starts
4. end, the character offset in the text where the annotation ends
5. Speaker, the xmi:id of the character reference that is used as speaker
6. SpokenTo, the xmi:id of the character reference that is used as addressee
7. Category, a string showing the category of this direct speech

Its is advised to start the parsing with the text, then continue with the character references and parse the direct speech annotations at the end.
Thie .xmi documents contain alongside the manually created annotations also the following automatically added annotations:

1. Chapter
2. Chunks
3. Dependency information
4. Morphology information
5. POS information
6. Sentences
7. Paragraphs

2) TEI-XML
-
Parsing TEI is much harder than the ".xmi", however we encoded the manually annotated features like this:

Within the <body> element (and an inner <p>-element) of each document, there is a list of tokens, encoded as <w>-elements. Those <w> elements all are represented by a unique id for each document.
A small example:
```html
<body>
<p>
<w xml:id="w1">In</w>
<w xml:id="w2">diesem</w>
<w xml:id="w3">Augenblick</w>
<w xml:id="w4">war</w>
<persName type="Core" xml:id="cr1" ><w xml:id="w5">Jakob</w>
<w xml:id="w6">Collin</w>
</persName>
<w xml:id="w7">bereits</w>
<w xml:id="w8">seit</w>
<w xml:id="w9">etwa</w>
<w xml:id="w10">einer</w>
<w xml:id="w11">halben</w>
<w xml:id="w12">Stunde</w>
<w xml:id="w13">mit</w>
<persName type="pron" xml:id="cr2" prev="#cr1"><w xml:id="w14">seiner</w>
</persName>
<w xml:id="w15">gründlichen</w>
<w xml:id="w16">Überlegung</w>
<w xml:id="w17">fertig</w>
```

Embedded around the tokens, there are <persName> elements, one for each character reference. They got the assigned type ("pron","core","apptdfw","appA") and a (optional) pointer to a previous <persName> element.
By following those pointers one can derive the clustering induced by the coreference (in fact all references point to the first appearance of the corresponding entity, transitively following of those edges is therefore unnecessary)


Accessing the direct speech can be done analogous to accessing the references. This time you neeed to search all <quote> elements inside the document

```html
<quote type="directspeech"><sp who="#cr87"/><w xml:id="w837">»</w>
<w xml:id="w838">Zum</w>
<w xml:id="w839">Trödelmarkt</w>
<w xml:id="w840">,</w>
<w xml:id="w841">und</w>
<w xml:id="w842">Karriere</w>
<w xml:id="w843">;</w>
<w xml:id="w844">es</w>
<w xml:id="w845">gibt</w>
<w xml:id="w846">was</w>
<w xml:id="w847">zu</w>
<w xml:id="w848">verdienen</w>
<w xml:id="w849">!</w>
<w xml:id="w850">«</w>
</quote>
```

Embedded in those quotes are speaker elements <sp> (optionally but ususally present) The type of the direct speech is one of (directspeech|thought|citation|fictionalspeech|name|other). The speaker element has a "who" attribute that points to xml:id of the speaking character reference.

On top, sentences and paragraphs have been added using virtual "<join>" elements. An example for a sentence from DROC:
```html
<join results="s" scope ="root" target="#w4735 #w4736 #w4737 #w4738 #w4739 #w4740 #w4741 #w4742 #w4743 #w4744 #w4745 #w4746 #w4747 #w4748 #w4749 #w4750 #w4751 #w4752 #w4753 #w4754 #w4755 #w4756 #w4757 #w4758 "/>
```

This indicates that the sentence starts at token with id "w4735" and ends at token with id "w4758".
Citing DROC
----

If you use DROC in your experiments please cite us, using:

Markus Krug, Frank Puppe, Isabella Reger, Lukas Weimer, Luisa Macharowsky, Stephan Feldhaus, Fotis Jannidis: Description of a Corpus of Character References in German Novels - DROC [Deutsches ROman Corpus]. DARIAH-DE Working Papers Nr. 27. Göttingen: DARIAH-DE, 2018. URN: urn:nbn:de:gbv:7-dariah-2018-2-9

Licensing
----
The corpus is licensed under the Creative Commons license CC-BY

