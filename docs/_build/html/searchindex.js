Search.setIndex({docnames:["gojo","gojo.core","gojo.deepl","gojo.experimental","gojo.interfaces","gojo.plotting","gojo.util","index","modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["gojo.rst","gojo.core.rst","gojo.deepl.rst","gojo.experimental.rst","gojo.interfaces.rst","gojo.plotting.rst","gojo.util.rst","index.rst","modules.rst"],objects:{"":[[0,0,0,"-","gojo"]],"gojo.core":[[1,0,0,"-","evaluation"],[1,0,0,"-","loops"],[1,0,0,"-","report"]],"gojo.core.evaluation":[[1,1,1,"","Metric"],[1,2,1,"","flatFunctionInput"],[1,2,1,"","getAvailableDefaultMetrics"],[1,2,1,"","getDefaultMetrics"],[1,2,1,"","getScores"]],"gojo.core.loops":[[1,2,1,"","evalCrossVal"],[1,2,1,"","evalCrossValNestedHPO"]],"gojo.core.report":[[1,1,1,"","CVReport"]],"gojo.core.report.CVReport":[[1,3,1,"","addMetadata"],[1,3,1,"","getFittedTransforms"],[1,3,1,"","getScores"],[1,3,1,"","getTestPredictions"],[1,3,1,"","getTrainPredictions"],[1,3,1,"","getTrainedModels"],[1,4,1,"","metadata"]],"gojo.deepl":[[2,0,0,"-","callback"],[2,0,0,"-","cnn"],[2,0,0,"-","ffn"],[2,0,0,"-","loading"],[2,0,0,"-","loops"],[2,0,0,"-","loss"],[2,0,0,"-","models"]],"gojo.deepl.callback":[[2,1,1,"","Callback"],[2,1,1,"","EarlyStopping"]],"gojo.deepl.callback.Callback":[[2,3,1,"","evaluate"],[2,3,1,"","resetState"]],"gojo.deepl.callback.EarlyStopping":[[2,5,1,"","DIRECTIVE"],[2,5,1,"","VALID_TRACKING_OPTS"],[2,3,1,"","evaluate"],[2,3,1,"","resetState"]],"gojo.deepl.cnn":[[2,1,1,"","ResNetBlock"]],"gojo.deepl.cnn.ResNetBlock":[[2,3,1,"","forward"]],"gojo.deepl.ffn":[[2,2,1,"","createSimpleFFNModel"],[2,2,1,"","createSimpleParametrizedFFNModel"]],"gojo.deepl.loading":[[2,1,1,"","GraphDataset"],[2,1,1,"","TorchDataset"]],"gojo.deepl.loops":[[2,2,1,"","fitNeuralNetwork"],[2,2,1,"","getAvailableIterationFunctions"],[2,2,1,"","iterSupervisedEpoch"],[2,2,1,"","iterUnsupervisedEpoch"]],"gojo.deepl.loss":[[2,1,1,"","BCELoss"],[2,1,1,"","ELBO"],[2,2,1,"","huberLossWithNaNs"],[2,2,1,"","mseLossWithNaNs"],[2,2,1,"","weightedBCE"],[2,2,1,"","weightedBCEwithNaNs"]],"gojo.deepl.loss.BCELoss":[[2,3,1,"","forward"]],"gojo.deepl.loss.ELBO":[[2,3,1,"","forward"]],"gojo.deepl.models":[[2,1,1,"","FusionModel"],[2,1,1,"","GNN"],[2,1,1,"","MultiTaskFFN"],[2,1,1,"","MultiTaskFFNv2"],[2,1,1,"","VanillaVAE"]],"gojo.deepl.models.FusionModel":[[2,3,1,"","encode"],[2,3,1,"","forward"]],"gojo.deepl.models.GNN":[[2,3,1,"","ffnModel"],[2,3,1,"","forward"],[2,3,1,"","fusionModel"],[2,3,1,"","gnnForward"],[2,3,1,"","graphPooling"]],"gojo.deepl.models.MultiTaskFFN":[[2,3,1,"","forward"]],"gojo.deepl.models.MultiTaskFFNv2":[[2,3,1,"","forward"]],"gojo.deepl.models.VanillaVAE":[[2,3,1,"","decode"],[2,3,1,"","encode"],[2,3,1,"","forward"],[2,3,1,"","reparametrize"],[2,3,1,"","sample"]],"gojo.interfaces":[[4,0,0,"-","data"],[4,0,0,"-","model"],[4,0,0,"-","transform"]],"gojo.interfaces.data":[[4,1,1,"","Dataset"]],"gojo.interfaces.data.Dataset":[[4,4,1,"","array_data"],[4,4,1,"","index_values"],[4,4,1,"","var_names"]],"gojo.interfaces.model":[[4,1,1,"","Model"],[4,1,1,"","ParametrizedTorchSKInterface"],[4,1,1,"","SklearnModelWrapper"],[4,1,1,"","TorchSKInterface"]],"gojo.interfaces.model.Model":[[4,3,1,"","copy"],[4,3,1,"","fitted"],[4,3,1,"","getParameters"],[4,4,1,"","is_fitted"],[4,4,1,"","parameters"],[4,3,1,"","performInference"],[4,3,1,"","reset"],[4,3,1,"","resetFit"],[4,3,1,"","train"],[4,3,1,"","update"],[4,3,1,"","updateParameters"]],"gojo.interfaces.model.ParametrizedTorchSKInterface":[[4,3,1,"","copy"],[4,3,1,"","getParameters"],[4,3,1,"","updateParameters"]],"gojo.interfaces.model.SklearnModelWrapper":[[4,3,1,"","copy"],[4,3,1,"","getParameters"],[4,4,1,"","model"],[4,3,1,"","performInference"],[4,3,1,"","reset"],[4,3,1,"","train"],[4,3,1,"","updateParameters"]],"gojo.interfaces.model.TorchSKInterface":[[4,3,1,"","copy"],[4,4,1,"","fitting_history"],[4,3,1,"","getParameters"],[4,4,1,"","model"],[4,4,1,"","num_params"],[4,3,1,"","performInference"],[4,3,1,"","reset"],[4,3,1,"","train"],[4,3,1,"","updateParameters"]],"gojo.interfaces.transform":[[4,1,1,"","SKLearnTransformWrapper"],[4,1,1,"","Transform"]],"gojo.interfaces.transform.SKLearnTransformWrapper":[[4,3,1,"","copy"],[4,3,1,"","fit"],[4,3,1,"","getParameters"],[4,3,1,"","reset"],[4,3,1,"","transform"],[4,3,1,"","updateParameters"]],"gojo.interfaces.transform.Transform":[[4,3,1,"","copy"],[4,3,1,"","fit"],[4,3,1,"","fitted"],[4,3,1,"","getParameters"],[4,4,1,"","is_fitted"],[4,3,1,"","reset"],[4,3,1,"","resetFit"],[4,3,1,"","transform"],[4,3,1,"","update"],[4,3,1,"","updateParameters"]],"gojo.plotting":[[5,0,0,"-","basic"],[5,0,0,"-","classification"]],"gojo.plotting.basic":[[5,2,1,"","barPlot"],[5,2,1,"","linePlot"],[5,2,1,"","scatterPlot"]],"gojo.plotting.classification":[[5,2,1,"","confusionMatrix"],[5,2,1,"","roc"]],"gojo.util":[[6,0,0,"-","io"],[6,0,0,"-","login"],[6,0,0,"-","splitter"],[6,0,0,"-","tools"],[6,0,0,"-","validation"]],"gojo.util.io":[[6,2,1,"","load"],[6,2,1,"","loadJson"],[6,2,1,"","pprint"],[6,2,1,"","saveJson"],[6,2,1,"","serialize"]],"gojo.util.login":[[6,1,1,"","Login"],[6,2,1,"","configureLogger"],[6,2,1,"","deactivate"],[6,2,1,"","isActive"]],"gojo.util.login.Login":[[6,5,1,"","logger_levels"]],"gojo.util.splitter":[[6,1,1,"","InstanceLevelKFoldSplitter"],[6,1,1,"","SimpleSplitter"],[6,2,1,"","getCrossValObj"]],"gojo.util.splitter.InstanceLevelKFoldSplitter":[[6,3,1,"","split"]],"gojo.util.splitter.SimpleSplitter":[[6,3,1,"","split"]],"gojo.util.tools":[[6,2,1,"","getNumModelParams"],[6,2,1,"","minMaxScaling"],[6,2,1,"","zscoresScaling"]],"gojo.util.validation":[[6,2,1,"","checkCallable"],[6,2,1,"","checkClass"],[6,2,1,"","checkInputType"],[6,2,1,"","checkIterable"],[6,2,1,"","checkMultiInputTypes"],[6,2,1,"","fileExists"],[6,2,1,"","pathExists"]],gojo:[[1,0,0,"-","core"],[2,0,0,"-","deepl"],[3,0,0,"-","experimental"],[4,0,0,"-","interfaces"],[5,0,0,"-","plotting"],[6,0,0,"-","util"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","function","Python function"],"3":["py","method","Python method"],"4":["py","property","Python property"],"5":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:function","3":"py:method","4":"py:property","5":"py:attribute"},terms:{"0":[1,2,4,5,6,7],"00":1,"001":[2,4],"05":2,"0x7fd7ca47b940":4,"0x7fd7ca4805e0":4,"1":[1,2,4,5,6,7],"10":[1,2,4,5,6],"100":[1,2,4,5],"1000":[1,4,7],"1011":5,"1016":5,"1018":5,"11":5,"12":5,"13":[4,5],"1312":2,"1321":2,"14":5,"149":2,"15":5,"16":[2,4],"1997":[1,4,6,7],"1d":2,"1e":2,"2":[1,2,4,5,6],"20":[2,4,6],"200":[4,5],"2014":2,"2016":2,"223":2,"25":4,"250":2,"2965":2,"2d":2,"3":[1,2,4,5,7],"30":2,"32":4,"334":2,"3898":2,"3d":2,"4":[2,5],"40":[1,2],"4343":2,"5":[1,2,4,5,7],"50":[2,4,7],"500":2,"6":5,"60":2,"6114":2,"7":5,"75":4,"77":2,"770":2,"778":2,"8":[1,4],"80":1,"94":2,"987":5,"99":2,"992":5,"abstract":[2,4],"boolean":2,"break":2,"case":[1,2,7],"class":[0,1,2,4,5,6],"default":[1,2,4,5,6],"do":[1,2],"export":6,"final":2,"float":[1,2,4,5,6],"function":[1,2,4,5,6],"import":[1,2,4,5,6,7],"int":[0,1,2,4,5,6,7],"new":[4,5],"return":[1,2,4,5,6],"true":[1,2,4,5,6,7],"try":0,"var":2,"while":[2,4],A:[2,5],And:4,At:7,By:[1,2,4,5],For:[1,4],IS:7,If:[1,2,4,5,6,7],In:[1,2,7],It:[1,4],One:2,The:[1,2,4,6,7],Their:2,These:[1,2,7],With:7,_:2,__:4,ab:2,about:[1,4],absolut:6,access:2,accord:[1,2,4,6],account:1,accur:7,accuraci:1,accuracy_scor:1,across:[2,7],activ:[2,4,6],activation_fn:2,ad:[2,7],adam:[2,4],adapt:[2,5],add:[1,2,6],add_auc_info:5,add_time_prefix:6,addit:4,addmetadata:1,adj_matric:2,adj_matrix:2,adjac:2,adjust:[2,4],affin:2,after:[2,4,7],agg_funct:1,aggreg:[1,2],all:[1,2,4,7],allow:[2,4,6],allow_nan:2,along:2,alpha:[2,4,5,7],also:[1,4],although:7,an:[0,1,2,4,6,7],ani:[1,2,4,7],appear:7,append:[2,4],appli:[0,1,2,4,6],approach:2,ar:[1,2,4,5,7],architectur:2,arg:[2,6],argument:[1,2,4],arrai:[0,1,2,4,6],arrang:2,array_data:4,arxiv:2,associ:[1,2,4,5,6],assum:2,astyp:[1,4,7],attempt:4,attribut:2,auc:[4,5],auto:2,autoencod:2,auxiliari:2,avail:[1,2,4],averag:[2,5,7],avoid:1,ax:5,axi:[1,2,5,7],axis_label_pad:5,axis_label_s:5,axis_tick_s:5,backend:6,balanc:1,bar:5,barplot:5,base:[0,1,2,4,6],baseestim:4,basesampl:1,basic:[0,2,4,6,7],batch:[2,4],batch_siz:[2,4],batchnorm1d:2,batchnorm:2,bay:2,bceloss:[2,4],been:[1,2,4,7],befor:[1,6],behavior:4,being:1,below:7,beta:2,between:[2,6],bia:[2,4],bin_threshold:[1,2,4,5,7],binar:[1,5],binari:2,binary_classif:[1,2,4,5,7],black:5,block:2,blue:5,bool:[1,2,4,5,6],both:[1,2],bound:2,bug:7,build:2,c0:5,c1:5,c2:5,cache_s:[1,4,7],calcul:[1,2,5],call:[1,2,4,6],callabl:[1,2,4,6],callback:[0,4,7],can:[1,2,4,5,6],capsiz:5,carr:2,cat:2,categor:1,cater:7,cd:7,cdot:2,center:[4,5],chang:[2,4,7],channel:2,characterist:2,check:[6,7],checkcal:6,checkclass:6,checkinputtyp:6,checkiter:6,checkmultiinputtyp:6,class_weight:[1,4,7],classic:2,classif:[0,1,2,4,7],clone:7,cm_font_siz:5,cmap:5,cnn:[0,7],code:[2,7],codifi:2,coef0:[1,4,7],color:5,colormap:5,column:[1,5],com:7,commonli:2,compar:2,compat:[1,6],complet:2,complex:7,compon:7,comprehens:7,comput:[1,2,5,7],concat:[1,5],concat_fn:2,concaten:2,confer:2,configur:[1,6],configurelogg:6,confus:5,confusionmatrix:5,connect:2,consist:2,constructor:4,consult:[1,4],contain:[1,2,4],content:7,contribut:7,control:[2,4,5,6],conv1:2,conv2:2,converg:[4,5],convert:6,convolut:2,copi:[1,4],core:[0,2,4,5,7],correct:7,correspond:[1,2,4,6],cost:1,count:2,cpu:[2,4],creat:[1,2,4,6,7],createsimpleffnmodel:[2,4],createsimpleparametrizedffnmodel:2,creation:[2,7],cross:[1,2,4,5,6,7],cuda:2,current:[1,2,4,6,7],current_devic:2,curv:5,customiz:7,cv:[1,4,5,6,7],cv_obj:[1,6],cv_report:[1,4,5,7],cvreport:[1,5],cx:5,dadta:2,dash:[4,5],data:[0,1,2,5,6,7],databatch:2,datafram:[1,2,4,5,6,7],dataload:[2,4],dataloader_class:4,dataloadererror:0,dataset:[1,2,4,5,6,7],dataset_class:4,deactiv:6,debug:2,decai:2,decim:[1,4,7],decis:1,declar:2,decod:2,decoder_in_dim:2,decoder_out:2,decomposit:[1,4,5],decreas:2,deep:[2,4],deepcopi:[1,4],deepl:[0,4,5,7],deepl_load:2,defin:[1,2,4,5],definit:[2,5],defualt:2,degre:[1,4,7],delta:2,deriv:2,describ:2,design:[2,4,7],detected_class:0,develop:7,deviat:[2,5],devic:[2,4],df:5,dict:[1,2,4,5,6],dictionari:[1,2,4,6],differ:[0,2,5,7],dim:2,dimens:[0,2],dimension:2,direct:2,directli:[1,2],disabl:6,displai:[4,5],distribut:[1,2],diverg:2,document:7,doe:[0,2,4],domain:7,dot:5,dpi:5,driven:6,drop:1,drop_last:4,dropout:2,dure:[2,4],e:[2,4,5,7],each:[1,2,5,6],earli:2,earlystop:2,easi:[1,2,4],easili:[2,4],edg:2,edge_index:2,effect:[1,2,4,5],effici:7,either:2,elbo:2,element:[1,2,4,6],elimin:2,elu:[2,4],emb:2,emb_feat:2,embed:2,encod:2,encoder_out_dim:2,end:2,engin:7,entir:7,entri:2,entropi:2,ep:2,epoch:[2,4,5],equip:7,err:[4,5,6],err_alpha:5,err_caps:5,err_lw:5,error:[2,4,5,6,7],estim:4,evalcrossv:[1,4,5,7],evalcrossvalnestedhpo:[1,4],evalu:[0,2,4,6,7],evid:2,exampl:[2,4,7],except:4,execut:[2,4],exist:6,expect:[0,2],expected_n_dim:0,expedit:7,experi:7,experiment:[0,2,7],exponenti:2,express:5,extract:2,extractor:2,f1:4,f1_score:[1,4],fail:0,fals:[1,2,4,5,6],fashion:6,featur:[2,6,7],feature_extractor:2,feature_rang:6,feed:2,fernandogagu:7,few:7,ffn:[0,4,7],ffn_model:2,ffnmodel:2,figsiz:5,figur:5,file:6,fileexist:6,filter:1,fine:7,first:[2,4,6,7],fit:[0,1,2,4],fit_transform:5,fitneuralnetwork:[2,4,5],fitted_transform:1,fitted_transforms_kei:1,fitting_histori:[2,4],flatfunctioninput:1,flatten:[1,2],fn:1,fold:[1,5,6,7],folder:7,follow:[1,2,4],font:5,fore:4,forget:4,form:2,forward:2,framework:[1,7],from:[0,1,2,4,5,6,7],func:2,function_kw:4,funtion:2,fuse:2,fusion:2,fusion_model:2,fusionmodel:2,g:5,gather:5,gaussian:2,gaussiannb:4,gener:[1,2,4,5,6],generating_fn:4,get:[1,4,5],getavailabledefaultmetr:1,getavailableiterationfunct:2,getcrossvalobj:[1,4,5,6,7],getdefaultmetr:[1,2,4,5,7],getfittedtransform:1,getnummodelparam:6,getparamet:4,getscor:[1,4,5,7],gettestpredict:[1,5],gettrainedmodel:1,gettrainpredict:[1,5],gf_param:4,gf_params__layer_dim:4,ggplot:5,gird:5,git:7,github:7,given:[1,2,4,6],gnn:2,gnn_model:2,gnnforward:2,go:2,gp_agg:2,gradient:2,graph:2,graph_dt:2,graphdataset:2,graphpool:2,greater:2,grid:5,grid_alpha:5,ground:[2,5],group:5,guideli:2,ha:[2,4,7],handler:6,hash:5,have:[1,2,4,5],he:2,hhmmss:6,hide:5,hide_legend:5,hide_xlabel:5,hide_ylabel:5,higher:1,histor:2,histori:[1,2,4],homogen:4,hot:1,how:2,hpo:1,hpo_n_trial:1,hpo_sampl:1,http:[1,2],huber:2,huberlosswithnan:2,hue:5,hue_map:5,hyparamet:4,hyperparamet:[1,2,4],i:[2,4,5],identifi:[1,5,6],ieee:2,ignor:[1,2,4,6],ignore_bin_threshold:1,ignore_input:2,imag:2,implement:[1,2,4,6,7],important:2,in_channel:2,in_feat:[2,4],in_featur:[2,4],in_var:0,includ:[1,6],incorpor:[2,4],incorrectnumberofclass:0,index:[1,2,4,7],index_valu:4,indic:[1,2,4,5,6],infer:[4,5],inference_dataloader_kw:4,inference_dataset_kw:4,info:6,inform:[1,2,4,5,6],init:2,init_layer_dim:2,initi:[2,4],inner:[1,2,4],inner_cv:1,inplac:[1,2],input:[0,1,2,4,5,6],input_n_dim:0,input_obj_nam:6,input_var:6,input_var_nam:6,insid:4,instanc:[1,2,4,6],instance_id:6,instancelevelkfoldsplitt:[1,6],instead:4,integr:[2,4],interact:4,interfac:[0,1,2,7],interfacebut:4,interfaz:[1,7],intern:[1,2,4],interpret:2,invalu:7,involv:1,io:[0,7],is_fit:4,isact:6,issu:[4,7],it_without_improv:2,item:2,iter:[1,2,4,6],iter_fn:[2,4],iter_fn_kw:4,itersupervisedepoch:[2,4],iterunsupervisedepoch:2,its:4,j:2,job:1,joblib:6,joblib_gzip:6,journei:7,json:6,k:2,kaiming_uniform_:2,kei:[1,2,4],kernel:[1,2,4,7],kernel_s:2,kigma:2,kld_weight:2,kullback:2,kwarg:[1,2,4],l1:2,l2:2,l:2,label:[1,4,5,6],languag:7,larg:1,larger:1,last:2,latent:2,latent_dim:2,later:7,latest:2,layer:2,layer_activ:[2,4],layer_dim:[2,4],layer_dropout:2,lead:7,learn:[2,4],least:[2,6],leav:[1,6],leaveoneout:[1,6],legend:5,legend_bbox_to_anchor:5,legend_po:[4,5],legend_s:5,leibler:2,len:6,length:2,level:[1,2,4,5,6],librari:[1,4,6,7],lienar:2,like:1,line:5,lineal:2,linear:[2,4],linearlay:[2,4],lineplot:[4,5],linewidth:5,list:[1,2,4,5,6],load:[0,1,4,5,6,7],load_win:[1,4,5,7],loader:0,loadjson:6,locat:7,log:2,logarithm:2,logger:6,logger_level:6,logic:2,login:[0,7],loguru:6,logvar:2,loocv:[1,6],loop:[0,4,7],loss:[0,4,5,7],loss_fn:2,loss_funct:4,lower:[2,5],lr:[2,4],ls:[4,5],lw:5,made:[1,4,5],mai:1,main:2,make:[0,2,4,6],maker_s:5,mandatori:4,map:5,marker:5,match:[0,2,6],matplotlib:5,matric:2,matrix:[2,5],matriz:5,max:[2,6],max_depth:1,max_sampl:1,max_width:2,maxim:1,maximum:[2,4,5],mean:[1,2,4,5,6,7],melt:5,merg:2,metadata:[1,4],meth:2,method:[1,2,4,5,6],metric:[1,2,4,5,7],metric_scor:1,min:[2,6],min_width:2,minim:1,minimum:[2,5],minmaxsc:6,miss:2,missingarraydimens:0,ml:7,model1:5,model1_df:5,model2:5,model2_df:5,model:[0,1,5,6,7],model_class:4,model_fit:4,model_histori:4,model_paramet:4,model_pr:5,model_predict:4,model_select:[1,4,6],modif:1,modifi:[2,5],modulelist:2,moment:7,momentum:2,more:[1,2,4,7],mp:[2,4],mpl:5,mse:2,mselosswithnan:2,msg:0,mtt_out:2,mu:2,mu_std:2,multi:[1,2],multiclass:[1,4],multipl:6,multitask_model:2,multitask_project:2,multitaskffn:2,multitaskffnv2:2,multivari:2,multt_batchnorm:2,multt_clf_activ:2,multt_dropout:2,multt_layer_activ:2,multt_layer_dim:2,multt_reg_activ:2,must:[1,2,4,5,6],must_exist:6,n:[2,5,6],n_clf_task:2,n_compon:[1,4,5],n_epoch:[2,4],n_feat:[2,6],n_fold:[1,5],n_fold_kei:1,n_job:[1,7],n_layer:2,n_node:2,n_node_feat:2,n_node_featur:2,n_reg_task:2,n_repeat:6,n_roc_point:5,n_sampl:[2,6],n_split:6,n_startup_tri:1,naive_bay:4,name:[1,2,4,5],natur:7,ndarrai:[1,2,4,6],necessari:[2,7],need:6,nest:1,network:2,neural:2,nn:[2,4,6],no_grad:2,node:2,node_feat:2,none:[0,1,2,4,5,6,7],normal:[2,5],note:[1,2,4,6],notebook:7,now:4,np:[1,2,4,6],num_param:4,number:[0,1,2,4,5,6],number_of_class:1,numpi:[1,2,4,6],obj:6,object:[1,2,4,6],objective_metr:1,objet:1,observ:6,obtain:6,offer:7,onc:4,one:[1,2,6],onli:[1,2],op_instance_arg:[1,2],opac:5,opcaiti:5,oper:[0,2],optim:[1,2,4],optimic:1,optimizer_class:[2,4],optimizer_kw:4,optimizer_param:2,option:[0,1,2,4,5,6],optuna:1,order:[2,5],org:[1,2],os:2,ot:5,other:[2,7],otherwis:4,out:[1,2,4,5,6,7],out_channel:2,out_feat:[2,4],out_featur:[2,4],outer_cv:1,outpu:2,output:[2,6],output_activ:[2,4],over:[1,2,7],overview:2,overwrit:6,own:4,p:2,packag:7,pad:[2,5],padding_conv1:2,padding_conv2:2,page:7,panda:[1,2,4,5,6,7],parallel:1,param:2,paramet:[1,2,4,5,6],parameter:[2,4],parametr:4,parametrizedtorchskinterfac:[2,4],paramt:4,partit:6,pass:[1,2,4],path:[4,6],pathexist:6,pattern:2,pc1:5,pc2:5,pc:5,pca:[1,4,5],pd:[1,2,4,5,6,7],percentag:[4,5],perform:[0,1,2,4,5,6,7],performance_metr:1,performinfer:[4,5],pip:7,pipelin:4,place:2,pleas:7,plot:[0,4,7],point:5,poli:[1,4,7],pool:2,poorli:7,posibl:2,posit:[2,5],possibl:[1,6,7],pp:2,pprint:6,pre:1,pred_label:5,pred_test_kei:1,pred_train_kei:1,predefin:0,predict:[0,1,2,4,5,7],predict_proba:4,predictor:[2,4],prefix:6,preprocess:[1,4,5,7],present:[2,5],prevent:2,previou:[0,1,2,4],previous:6,print:[2,6],prior:4,problem:[1,2,4,7],proceed:2,process:[2,7],project:[2,7],properti:[1,4],proport:6,provid:[1,2,4,5,6,7],py:7,pyplot:5,python:[6,7],pytorch:[2,4],quick:2,rais:[1,4],rand:2,randint:[2,6],random:[2,4,5,6],random_color:5,random_l:5,random_label:5,random_lw:5,random_st:[1,4,6,7],rang:[2,6,7],rate:[2,5],raw_result:1,re:7,receiv:[1,2,4],recogn:4,recognit:2,recommend:7,reconstruct:2,redirect:6,regress:[1,2],rel:5,relat:2,releg:1,relu:2,remain:2,ren:2,reparametr:2,repeat:[1,6],repeatedkfold:[1,6],repeatedstratifiedkfold:[1,6],repetit:6,replic:6,report1:5,report:[0,5,7],repositori:7,repres:[1,4,5,6],represent:[2,5],requir:2,research:7,reset:[2,4],resetfit:4,resetst:2,residu:2,resnetblock:2,resourc:7,respect:2,rest:[1,4,6,7],result:[1,2,5],review:7,rich:7,right:[4,5],robust:7,roc:5,rotat:5,round:[1,4,7],run:[2,7],s:[2,5],same:[1,2,6],sampl:[1,2,6],sampler:1,save:[1,5,6],save_kw:5,save_model:1,save_train_pr:1,save_transform:1,savejson:6,scaffold:2,scalar:[1,2],scale:[2,6],scaled_data:6,scatter:5,scatterplot:5,schema:1,scheme:[1,6],scientist:7,score:[1,4,6,7],scores_1:5,scores_2:5,script:7,search:[1,7],search_spac:1,second:[2,4,6],see:[1,2,4,5,6,7],seed:[4,6],seen:4,select:[1,2,4],sep:6,separ:[4,5],sequenti:[2,4],seri:[1,2,4,6],serial:6,serv:7,set:[1,2,4,5,6,7],setup:7,sever:2,shadow:5,shape:[2,4,5],should:[1,2,4,6],show:[2,5],show_random:5,shown:7,shuffl:[2,4,6],sigmoid:[2,4],silenc:1,similar:2,simpl:[1,2,6],simplesplitt:[1,6],simplifi:[2,7],singl:[2,5],size:[2,4,5,6],sklearn:[1,4,5,6,7],sklearnmodelwrapp:[1,4,7],sklearntransformwrapp:[1,4],solid:[4,5],solut:7,some:[1,4],sourc:[0,1,2,4,5,6],space:[1,2],special:2,specif:7,specifi:[1,2,4,5,6],specified_class:0,split:[4,6],splitter:[0,1,7],squar:2,stabil:7,standar:4,standard:[2,4,5,6],standardscal:[1,4,5],state:[2,6],statist:4,statu:6,std:[2,4,5],std_x:4,still:7,stop:2,storag:1,store:[2,4],str:[0,1,2,4,5,6],strategi:1,stratif:[4,6],stratifi:[1,4,5,6,7],stream:2,streamlin:7,stride:2,stride_conv1:2,stride_conv2:2,string:[1,2,5],structur:7,studi:6,style:5,subclass:[2,4],subject:7,submodul:7,subroutin:[1,4],subsequ:[2,4],success:6,suggest:1,suggest_categor:1,suggest_float:1,suggest_int:1,suit:7,sum:2,sun:2,superclass:4,supervis:2,support:2,suppress:4,supress:1,supress_warn:[1,4],svc:[1,4,7],svm:[1,4,7],sy:4,system:[6,7],t:4,tabular:2,tabular_x:2,take:[1,2],tanh:2,tanu:2,target:[1,2,4,5,7],target_nam:[1,4],task:[1,2,7],task_info:1,tensor:2,term:2,test:[1,4,5,6,7],test_idx:6,test_idx_kei:1,test_info:5,test_predict:1,test_scor:7,test_siz:6,than:2,thei:[1,5,7],them:7,thi:[1,2,4,5,6,7],those:1,threshold:[1,2,5],through:2,thrown:0,tick:5,time:6,time_prefix:6,titl:[4,5],title_pad:5,title_s:5,togeth:4,tool:[0,7],torch:[2,4,6],torch_geometr:2,torchdataset:[2,4],torchskinterfac:[2,4],tpesampl:1,track:2,track_running_stat:2,train:[1,2,4,5,6,7],train_dataloader_kw:4,train_dataloader_kw__batch_s:4,train_dataset_kw:4,train_dl:2,train_idx:6,train_idx_kei:1,train_info:5,train_loss:2,train_metr:2,train_siz:4,train_split:4,train_split_stratifi:4,train_test_split:[4,6],trainabl:[4,6],trained_model:1,trained_model_kei:1,transform:[0,1,2,7],transform_class:4,trick:2,tridimension:2,true_label:5,true_test_kei:1,true_train_kei:1,truth:5,tune:7,tupl:[2,4,5,6],two:[1,2],type:[1,2,4,6],typic:[2,4,7],ultim:7,under:7,undetect:7,unfittedestim:[0,4],unfittedtransform:0,unifi:2,uniform:[1,2,6],union:2,until:2,updat:[2,4],updateparamet:4,upper:5,us:[1,2,4,5,6,7],use_multiclass_spars:1,user:[0,1,4,5],usual:4,util:[0,1,2,4,5,7],valid:[0,1,2,4,5,7],valid_dataloader_kw:4,valid_dataset_kw:4,valid_dl:2,valid_info:5,valid_loss:2,valid_metr:2,valid_tracking_opt:2,valid_typ:6,valu:[1,2,4,5],vanillava:2,var_nam:4,variabl:[1,2,4,5,6,7],variat:2,varieti:7,variou:4,vector:[1,2],verbos:[1,2,4,6],version:[2,4,7],via:7,vision:[2,7],vs:[1,4,7],wa:[4,5,6],want:[1,4],warn:[1,4,6],weight:[2,4],weightedbc:2,weightedbcewithnan:2,weights_init:2,well:2,were:1,when:[0,1,4,6],where:[1,2,6,7],whether:[1,2,4,5,6,7],which:[1,2,5],whose:[1,2],wide:7,width:[2,5],wine:[1,4,5,7],wine_dt:[1,4,5,7],within:[1,6],without:[0,2],work:7,would:2,wrapper:[2,4,6],x:[1,2,4,5,6,7],x_dataset:1,x_hat:2,x_loading_fn:2,x_new:4,x_stream_data:2,x_test:5,x_train:[4,5],x_tran:4,x_transform:2,x_true:2,x_valid:4,xaxis_label:5,xaxis_rot:5,xaxis_tick_s:5,xlabel_s:5,xs:4,xvmax:5,xvmin:5,xy:5,y:[1,2,4,5,6,7],y_dataset:1,y_hat:[2,4],y_loading_fn:2,y_pred:[1,4,5],y_pred_threshold:5,y_preds1:5,y_preds2:5,y_stream_data:2,y_test:5,y_train:[4,5],y_transform:2,y_true:[1,2,4,5],y_valid:4,yaxis_label:5,yaxis_rot:5,yaxis_tick_s:5,ylabel_s:5,you:[1,4,7],your:7,ys:4,yvmax:5,yvmin:5,yyyymmdd:6,z:[2,6],zhang:2,zscoressc:[4,6]},titles:["gojo package","gojo.core package","gojo.deepl package","gojo.experimental package","gojo.interfaces package","gojo.plotting package","gojo.util package","GoJo - Make Machine/Deep Learning pipelines simple","gojo"],titleterms:{api:7,basic:5,callback:2,classif:5,cnn:2,content:[0,1,2,3,4,5,6],core:1,data:4,deep:7,deepl:2,evalu:1,except:0,experiment:3,ffn:2,gojo:[0,1,2,3,4,5,6,7,8],indic:7,instal:7,interfac:4,io:6,learn:7,load:2,login:6,loop:[1,2],loss:2,machin:7,make:7,model:[2,4],modul:[0,1,2,3,4,5,6,7],packag:[0,1,2,3,4,5,6],pipelin:7,plot:5,report:1,simpl:7,splitter:6,submodul:[0,1,2,4,5,6],subpackag:0,tabl:7,tool:6,transform:4,usag:7,util:6,valid:6,warranti:7}})