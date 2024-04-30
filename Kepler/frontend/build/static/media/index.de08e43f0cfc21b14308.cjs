"use strict";var __defProp=Object.defineProperty,__getOwnPropDesc=Object.getOwnPropertyDescriptor,__getOwnPropNames=Object.getOwnPropertyNames,__hasOwnProp=Object.prototype.hasOwnProperty,__export=(e,t)=>{for(var s in t)__defProp(e,s,{get:t[s],enumerable:!0})},__copyProps=(e,t,s,r)=>{if(t&&"object"===typeof t||"function"===typeof t)for(let o of __getOwnPropNames(t))__hasOwnProp.call(e,o)||o===s||__defProp(e,o,{get:()=>t[o],enumerable:!(r=__getOwnPropDesc(t,o))||r.enumerable});return e},__toCommonJS=e=>__copyProps(__defProp({},"__esModule",{value:!0}),e),src_exports={};__export(src_exports,{JSONLoader:()=>JSONLoader,JSONWriter:()=>JSONWriter,NDJSONLoader:()=>NDJSONLoader,_ClarinetParser:()=>ClarinetParser,_GeoJSONLoader:()=>GeoJSONLoader,_GeoJSONWorkerLoader:()=>GeoJSONWorkerLoader,_GeoJSONWriter:()=>GeoJSONWriter,_JSONPath:()=>JSONPath,_rebuildJsonObject:()=>rebuildJsonObject}),module.exports=__toCommonJS(src_exports);var import_schema=require("@loaders.gl/schema");function parseJSONSync(e,t){var s;try{const r=JSON.parse(e);if(null==(s=t.json)?void 0:s.table){const e=getFirstArray(r)||r;return(0,import_schema.makeTableFromData)(e)}return r}catch(r){throw new Error("JSONLoader: failed to parse JSON")}}function getFirstArray(e){if(Array.isArray(e))return e;if(e&&"object"===typeof e)for(const t of Object.values(e)){const e=getFirstArray(t);if(e)return e}return null}var import_schema2=require("@loaders.gl/schema"),import_loader_utils=require("@loaders.gl/loader-utils"),MAX_BUFFER_LENGTH=Number.MAX_SAFE_INTEGER,Char={tab:9,lineFeed:10,carriageReturn:13,space:32,doubleQuote:34,plus:43,comma:44,minus:45,period:46,_0:48,_9:57,colon:58,E:69,openBracket:91,backslash:92,closeBracket:93,a:97,b:98,e:101,f:102,l:108,n:110,r:114,s:115,t:116,u:117,openBrace:123,closeBrace:125},stringTokenPattern=/[\\"\n]/g,DEFAULT_OPTIONS={onready:()=>{},onopenobject:()=>{},onkey:()=>{},oncloseobject:()=>{},onopenarray:()=>{},onclosearray:()=>{},onvalue:()=>{},onerror:()=>{},onend:()=>{},onchunkparsed:()=>{}},ClarinetParser=class{constructor(e={}){this.options=DEFAULT_OPTIONS,this.bufferCheckPosition=MAX_BUFFER_LENGTH,this.q="",this.c="",this.p="",this.closed=!1,this.closedRoot=!1,this.sawRoot=!1,this.error=null,this.state=0,this.stack=[],this.position=0,this.column=0,this.line=1,this.slashed=!1,this.unicodeI=0,this.unicodeS=null,this.depth=0,this.options={...DEFAULT_OPTIONS,...e},this.textNode=void 0,this.numberNode="",this.emit("onready")}end(){return 1===this.state&&0===this.depth||this._error("Unexpected end"),this._closeValue(),this.c="",this.closed=!0,this.emit("onend"),this}resume(){return this.error=null,this}close(){return this.write(null)}emit(e,t){var s,r;null==(r=(s=this.options)[e])||r.call(s,t,this)}emitNode(e,t){this._closeValue(),this.emit(e,t)}write(e){if(this.error)throw this.error;if(this.closed)return this._error("Cannot write after close. Assign an onready handler.");if(null===e)return this.end();let t=0,s=e.charCodeAt(0),r=this.p;for(;s&&(r=s,this.c=s=e.charCodeAt(t++),r!==s?this.p=r:r=this.p,s);)switch(this.position++,s===Char.lineFeed?(this.line++,this.column=0):this.column++,this.state){case 0:s===Char.openBrace?this.state=2:s===Char.openBracket?this.state=4:isWhitespace(s)||this._error("Non-whitespace before {[.");continue;case 10:case 2:if(isWhitespace(s))continue;if(10===this.state)this.stack.push(11);else{if(s===Char.closeBrace){this.emit("onopenobject"),this.depth++,this.emit("oncloseobject"),this.depth--,this.state=this.stack.pop()||1;continue}this.stack.push(3)}s===Char.doubleQuote?this.state=7:this._error('Malformed object key should start with "');continue;case 11:case 3:if(isWhitespace(s))continue;s===Char.colon?(3===this.state?(this.stack.push(3),this._closeValue("onopenobject"),this.depth++):this._closeValue("onkey"),this.state=1):s===Char.closeBrace?(this.emitNode("oncloseobject"),this.depth--,this.state=this.stack.pop()||1):s===Char.comma?(3===this.state&&this.stack.push(3),this._closeValue(),this.state=10):this._error("Bad object");continue;case 4:case 1:if(isWhitespace(s))continue;if(4===this.state){if(this.emit("onopenarray"),this.depth++,this.state=1,s===Char.closeBracket){this.emit("onclosearray"),this.depth--,this.state=this.stack.pop()||1;continue}this.stack.push(5)}s===Char.doubleQuote?this.state=7:s===Char.openBrace?this.state=2:s===Char.openBracket?this.state=4:s===Char.t?this.state=12:s===Char.f?this.state=15:s===Char.n?this.state=19:s===Char.minus?this.numberNode+="-":Char._0<=s&&s<=Char._9?(this.numberNode+=String.fromCharCode(s),this.state=23):this._error("Bad value");continue;case 5:if(s===Char.comma)this.stack.push(5),this._closeValue("onvalue"),this.state=1;else if(s===Char.closeBracket)this.emitNode("onclosearray"),this.depth--,this.state=this.stack.pop()||1;else{if(isWhitespace(s))continue;this._error("Bad array")}continue;case 7:void 0===this.textNode&&(this.textNode="");let o=t-1,n=this.slashed,a=this.unicodeI;e:for(;;){for(;a>0;)if(this.unicodeS+=String.fromCharCode(s),s=e.charCodeAt(t++),this.position++,4===a?(this.textNode+=String.fromCharCode(parseInt(this.unicodeS,16)),a=0,o=t-1):a++,!s)break e;if(s===Char.doubleQuote&&!n){this.state=this.stack.pop()||1,this.textNode+=e.substring(o,t-1),this.position+=t-1-o;break}if(s===Char.backslash&&!n&&(n=!0,this.textNode+=e.substring(o,t-1),this.position+=t-1-o,s=e.charCodeAt(t++),this.position++,!s))break;if(n){if(n=!1,s===Char.n?this.textNode+="\n":s===Char.r?this.textNode+="\r":s===Char.t?this.textNode+="\t":s===Char.f?this.textNode+="\f":s===Char.b?this.textNode+="\b":s===Char.u?(a=1,this.unicodeS=""):this.textNode+=String.fromCharCode(s),s=e.charCodeAt(t++),this.position++,o=t-1,s)continue;break}stringTokenPattern.lastIndex=t;const r=stringTokenPattern.exec(e);if(null===r){t=e.length+1,this.textNode+=e.substring(o,t-1),this.position+=t-1-o;break}if(t=r.index+1,s=e.charCodeAt(r.index),!s){this.textNode+=e.substring(o,t-1),this.position+=t-1-o;break}}this.slashed=n,this.unicodeI=a;continue;case 12:s===Char.r?this.state=13:this._error(`Invalid true started with t${s}`);continue;case 13:s===Char.u?this.state=14:this._error(`Invalid true started with tr${s}`);continue;case 14:s===Char.e?(this.emit("onvalue",!0),this.state=this.stack.pop()||1):this._error(`Invalid true started with tru${s}`);continue;case 15:s===Char.a?this.state=16:this._error(`Invalid false started with f${s}`);continue;case 16:s===Char.l?this.state=17:this._error(`Invalid false started with fa${s}`);continue;case 17:s===Char.s?this.state=18:this._error(`Invalid false started with fal${s}`);continue;case 18:s===Char.e?(this.emit("onvalue",!1),this.state=this.stack.pop()||1):this._error(`Invalid false started with fals${s}`);continue;case 19:s===Char.u?this.state=20:this._error(`Invalid null started with n${s}`);continue;case 20:s===Char.l?this.state=21:this._error(`Invalid null started with nu${s}`);continue;case 21:s===Char.l?(this.emit("onvalue",null),this.state=this.stack.pop()||1):this._error(`Invalid null started with nul${s}`);continue;case 22:s===Char.period?(this.numberNode+=".",this.state=23):this._error("Leading zero not followed by .");continue;case 23:Char._0<=s&&s<=Char._9?this.numberNode+=String.fromCharCode(s):s===Char.period?(-1!==this.numberNode.indexOf(".")&&this._error("Invalid number has two dots"),this.numberNode+="."):s===Char.e||s===Char.E?(-1===this.numberNode.indexOf("e")&&-1===this.numberNode.indexOf("E")||this._error("Invalid number has two exponential"),this.numberNode+="e"):s===Char.plus||s===Char.minus?(r!==Char.e&&r!==Char.E&&this._error("Invalid symbol in number"),this.numberNode+=String.fromCharCode(s)):(this._closeNumber(),t--,this.state=this.stack.pop()||1);continue;default:this._error(`Unknown state: ${this.state}`)}return this.position>=this.bufferCheckPosition&&checkBufferLength(this),this.emit("onchunkparsed"),this}_closeValue(e="onvalue"){void 0!==this.textNode&&this.emit(e,this.textNode),this.textNode=void 0}_closeNumber(){this.numberNode&&this.emit("onvalue",parseFloat(this.numberNode)),this.numberNode=""}_error(e=""){this._closeValue(),e+=`\nLine: ${this.line}\nColumn: ${this.column}\nChar: ${this.c}`;const t=new Error(e);this.error=t,this.emit("onerror",t)}};function isWhitespace(e){return e===Char.carriageReturn||e===Char.lineFeed||e===Char.space||e===Char.tab}function checkBufferLength(e){const t=Math.max(MAX_BUFFER_LENGTH,10);let s=0;for(const r of["textNode","numberNode"]){const o=void 0===e[r]?0:e[r].length;if(o>t)if("text"===r);else e._error(`Max buffer length exceeded: ${r}`);s=Math.max(s,o)}e.bufferCheckPosition=MAX_BUFFER_LENGTH-s+e.position}var JSONPath=class{constructor(e=null){if(this.path=["$"],e instanceof JSONPath)this.path=[...e.path];else if(Array.isArray(e))this.path.push(...e);else if("string"===typeof e&&(this.path=e.split("."),"$"!==this.path[0]))throw new Error("JSONPaths must start with $")}clone(){return new JSONPath(this)}toString(){return this.path.join(".")}push(e){this.path.push(e)}pop(){return this.path.pop()}set(e){this.path[this.path.length-1]=e}equals(e){if(!this||!e||this.path.length!==e.path.length)return!1;for(let t=0;t<this.path.length;++t)if(this.path[t]!==e.path[t])return!1;return!0}setFieldAtPath(e,t){const s=[...this.path];s.shift();const r=s.pop();for(const o of s)e=e[o];e[r]=t}getFieldAtPath(e){const t=[...this.path];t.shift();const s=t.pop();for(const r of t)e=e[r];return e[s]}},JSONParser=class{constructor(e){this.result=void 0,this.previousStates=[],this.currentState=Object.freeze({container:[],key:null}),this.jsonpath=new JSONPath,this.reset(),this.parser=new ClarinetParser({onready:()=>{this.jsonpath=new JSONPath,this.previousStates.length=0,this.currentState.container.length=0},onopenobject:e=>{this._openObject({}),"undefined"!==typeof e&&this.parser.emit("onkey",e)},onkey:e=>{this.jsonpath.set(e),this.currentState.key=e},oncloseobject:()=>{this._closeObject()},onopenarray:()=>{this._openArray()},onclosearray:()=>{this._closeArray()},onvalue:e=>{this._pushOrSet(e)},onerror:e=>{throw e},onend:()=>{this.result=this.currentState.container.pop()},...e})}reset(){this.result=void 0,this.previousStates=[],this.currentState=Object.freeze({container:[],key:null}),this.jsonpath=new JSONPath}write(e){this.parser.write(e)}close(){this.parser.close()}_pushOrSet(e){const{container:t,key:s}=this.currentState;null!==s?(t[s]=e,this.currentState.key=null):t.push(e)}_openArray(e=[]){this.jsonpath.push(null),this._pushOrSet(e),this.previousStates.push(this.currentState),this.currentState={container:e,isArray:!0,key:null}}_closeArray(){this.jsonpath.pop(),this.currentState=this.previousStates.pop()}_openObject(e={}){this.jsonpath.push(null),this._pushOrSet(e),this.previousStates.push(this.currentState),this.currentState={container:e,isArray:!1,key:null}}_closeObject(){this.jsonpath.pop(),this.currentState=this.previousStates.pop()}},StreamingJSONParser=class extends JSONParser{constructor(e={}){super({onopenarray:()=>{if(!this.streamingArray&&this._matchJSONPath())return this.streamingJsonPath=this.getJsonPath().clone(),this.streamingArray=[],void this._openArray(this.streamingArray);this._openArray()},onopenobject:e=>{this.topLevelObject?this._openObject({}):(this.topLevelObject={},this._openObject(this.topLevelObject)),"undefined"!==typeof e&&this.parser.emit("onkey",e)}}),this.streamingJsonPath=null,this.streamingArray=null,this.topLevelObject=null;const t=e.jsonpaths||[];this.jsonPaths=t.map((e=>new JSONPath(e)))}write(e){super.write(e);let t=[];return this.streamingArray&&(t=[...this.streamingArray],this.streamingArray.length=0),t}getPartialResult(){return this.topLevelObject}getStreamingJsonPath(){return this.streamingJsonPath}getStreamingJsonPathAsString(){return this.streamingJsonPath&&this.streamingJsonPath.toString()}getJsonPath(){return this.jsonpath}_matchJSONPath(){const e=this.getJsonPath();if(0===this.jsonPaths.length)return!0;for(const t of this.jsonPaths)if(t.equals(e))return!0;return!1}};async function*parseJSONInBatches(e,t){var s;const r=(0,import_loader_utils.makeTextDecoderIterator)(e),{metadata:o}=t,{jsonpaths:n}=t.json||{};let a=!0;const i=new import_schema2.TableBatchBuilder(null,t),h=new StreamingJSONParser({jsonpaths:n});for await(const u of r){const e=h.write(u),r=e.length>0&&h.getStreamingJsonPathAsString();if(e.length>0&&a){if(o){const e={shape:(null==(s=null==t?void 0:t.json)?void 0:s.shape)||"array-row-table",batchType:"partial-result",data:[],length:0,bytesUsed:0,container:h.getPartialResult(),jsonpath:r};yield e}a=!1}for(const t of e){i.addRow(t);const e=i.getFullBatch({jsonpath:r});e&&(yield e)}i.chunkComplete(u);const n=i.getFullBatch({jsonpath:r});n&&(yield n)}const c=h.getStreamingJsonPathAsString(),l=i.getFinalBatch({jsonpath:c});if(l&&(yield l),o){const e={shape:"json",batchType:"final-result",container:h.getPartialResult(),jsonpath:h.getStreamingJsonPathAsString(),data:[],length:0};yield e}}function rebuildJsonObject(e,t){if((0,import_loader_utils.assert)("final-result"===e.batchType),"$"===e.jsonpath)return t;if(e.jsonpath&&e.jsonpath.length>1){const s=e.container;return new JSONPath(e.jsonpath).setFieldAtPath(s,t),s}return e.container}var VERSION="undefined"!==typeof __VERSION__?__VERSION__:"latest",JSONLoader={name:"JSON",id:"json",module:"json",version:VERSION,extensions:["json","geojson"],mimeTypes:["application/json"],category:"table",text:!0,options:{json:{shape:void 0,table:!1,jsonpaths:[]}},parse:parse,parseTextSync:parseTextSync,parseInBatches:parseInBatches};async function parse(e,t){return parseTextSync((new TextDecoder).decode(e),t)}function parseTextSync(e,t){return parseJSONSync(e,{...t,json:{...JSONLoader.options.json,...null==t?void 0:t.json}})}function parseInBatches(e,t){return parseJSONInBatches(e,{...t,json:{...JSONLoader.options.json,...null==t?void 0:t.json}})}var import_schema3=require("@loaders.gl/schema");function parseNDJSONSync(e){const t=e.trim().split("\n").map(((e,t)=>{try{return JSON.parse(e)}catch(s){throw new Error(`NDJSONLoader: failed to parse JSON on line ${t+1}`)}}));return(0,import_schema3.makeTableFromData)(t)}var import_schema4=require("@loaders.gl/schema"),import_loader_utils2=require("@loaders.gl/loader-utils");async function*parseNDJSONInBatches(e,t){const s=(0,import_loader_utils2.makeTextDecoderIterator)(e),r=(0,import_loader_utils2.makeLineIterator)(s),o=(0,import_loader_utils2.makeNumberedLineIterator)(r),n=new import_schema4.TableBatchBuilder(null,{...t,shape:"row-table"});for await(const{counter:h,line:c}of o)try{const e=JSON.parse(c);n.addRow(e),n.chunkComplete(c);const t=n.getFullBatch();t&&(yield t)}catch(i){throw new Error(`NDJSONLoader: failed to parse JSON on line ${h}`)}const a=n.getFinalBatch();a&&(yield a)}var VERSION2="undefined"!==typeof __VERSION__?__VERSION__:"latest",NDJSONLoader={name:"NDJSON",id:"ndjson",module:"json",version:VERSION2,extensions:["ndjson","jsonl"],mimeTypes:["application/x-ndjson","application/jsonlines","application/json-seq"],category:"table",text:!0,parse:async e=>parseNDJSONSync((new TextDecoder).decode(e)),parseTextSync:parseNDJSONSync,parseInBatches:parseNDJSONInBatches,options:{}},import_schema5=require("@loaders.gl/schema");function encodeTableAsJSON(e,t){var s;const r=(null==(s=null==t?void 0:t.json)?void 0:s.shape)||"object-row-table",o=[],n=(0,import_schema5.makeRowIterator)(e,r);for(const a of n)o.push(JSON.stringify(a));return`[${o.join(",")}]`}var JSONWriter={id:"json",version:"latest",module:"json",name:"JSON",extensions:["json"],mimeTypes:["application/json"],options:{},text:!0,encode:async(e,t)=>(new TextEncoder).encode(encodeTableAsJSON(e,t)).buffer,encodeTextSync:(e,t)=>encodeTableAsJSON(e,t)},import_gis=require("@loaders.gl/gis"),VERSION3="undefined"!==typeof __VERSION__?__VERSION__:"latest",GeoJSONWorkerLoader={name:"GeoJSON",id:"geojson",module:"geojson",version:VERSION3,worker:!0,extensions:["geojson"],mimeTypes:["application/geo+json"],category:"geometry",text:!0,options:{geojson:{shape:"object-row-table"},json:{shape:"object-row-table",jsonpaths:["$","$.features"]},gis:{format:"geojson"}}},GeoJSONLoader={...GeoJSONWorkerLoader,parse:parse2,parseTextSync:parseTextSync2,parseInBatches:parseInBatches2};async function parse2(e,t){return parseTextSync2((new TextDecoder).decode(e),t)}function parseTextSync2(e,t){let s;(t={...GeoJSONLoader.options,...t}).geojson={...GeoJSONLoader.options.geojson,...t.geojson},t.gis=t.gis||{};try{s=JSON.parse(e)}catch{s={}}const r={shape:"geojson-table",type:"FeatureCollection",features:(null==s?void 0:s.features)||[]};return"binary"===t.gis.format?(0,import_gis.geojsonToBinary)(r.features):r}function parseInBatches2(e,t){(t={...GeoJSONLoader.options,...t}).json={...GeoJSONLoader.options.geojson,...t.geojson};const s=parseJSONInBatches(e,t);return"binary"===t.gis.format?makeBinaryGeometryIterator(s):s}async function*makeBinaryGeometryIterator(e){for await(const t of e)t.data=(0,import_gis.geojsonToBinary)(t.data),yield t}var import_loader_utils3=require("@loaders.gl/loader-utils"),import_schema8=require("@loaders.gl/schema"),import_schema6=require("@loaders.gl/schema");function detectGeometryColumnIndex(e){var t;const s=(null==(t=e.schema)?void 0:t.fields.findIndex((e=>"geometry"===e.name)))??-1;if(s>-1)return s;if((0,import_schema6.getTableLength)(e)>0){const t=(0,import_schema6.getTableRowAsArray)(e,0);for(let s=0;s<(0,import_schema6.getTableNumCols)(e);s++){const e=null==t?void 0:t[s];if(e&&"object"===typeof e)return s}}throw new Error("Failed to detect geometry column")}function getRowPropertyObject(e,t,s=[]){var r;const o={};for(let n=0;n<(0,import_schema6.getTableNumCols)(e);++n){const a=null==(r=e.schema)?void 0:r.fields[n].name;a&&!s.includes(n)&&(o[a]=t[a])}return o}var import_schema7=require("@loaders.gl/schema");function encodeTableRow(e,t,s,r){const o=(0,import_schema7.getTableRowAsObject)(e,t);if(!o)return;const n=getFeatureFromRow(e,o,s),a=JSON.stringify(n);r.push(a)}function getFeatureFromRow(e,t,s){var r;const o=getRowPropertyObject(e,t,[s]),n=null==(r=e.schema)?void 0:r.fields[s].name;let a=n&&t[n];if(!a)return{type:"Feature",geometry:null,properties:o};if("string"===typeof a)try{a=JSON.parse(a)}catch(i){throw new Error("Invalid string geometry")}if("object"!==typeof a||"string"!==typeof(null==a?void 0:a.type))throw new Error("invalid geometry column value");return"Feature"===(null==a?void 0:a.type)?{...a,properties:o}:{type:"Feature",geometry:a,properties:o}}var Utf8ArrayBufferEncoder=class{constructor(e){this.strings=[],this.totalLength=0,this.textEncoder=new TextEncoder,this.chunkSize=e}push(...e){for(const t of e)this.strings.push(t),this.totalLength+=t.length}isFull(){return this.totalLength>=this.chunkSize}getArrayBufferBatch(){return this.textEncoder.encode(this.getStringBatch()).buffer}getStringBatch(){const e=this.strings.join("");return this.strings=[],this.totalLength=0,e}};async function*encodeTableAsGeojsonInBatches(e,t={}){const s={geojson:{},chunkSize:1e4,...t},r=new Utf8ArrayBufferEncoder(s.chunkSize);s.geojson.featureArray||r.push("{\n",'"type": "FeatureCollection",\n','"features":\n'),r.push("[");let o=s.geojson.geometryColumn,n=!0,a=0;for await(const i of e){const e=a+(0,import_schema8.getTableLength)(i);o||(o=o||detectGeometryColumnIndex(i));for(let s=a;s<e;++s)n||r.push(","),r.push("\n"),n=!1,encodeTableRow(i,s,o,r),r.isFull()&&(yield r.getArrayBufferBatch()),a=e;const t=r.getArrayBufferBatch();t.byteLength>0&&(yield t)}r.push("\n"),r.push("]\n"),s.geojson.featureArray||r.push("}"),yield r.getArrayBufferBatch()}var GeoJSONWriter={id:"geojson",version:"latest",module:"geojson",name:"GeoJSON",extensions:["geojson"],mimeTypes:["application/geo+json"],text:!0,options:{geojson:{featureArray:!1,geometryColumn:null}},async encode(e,t){const s=encodeTableAsGeojsonInBatches([e],t);return await(0,import_loader_utils3.concatenateArrayBuffersAsync)(s)},encodeInBatches:(e,t)=>encodeTableAsGeojsonInBatches(e,t)};