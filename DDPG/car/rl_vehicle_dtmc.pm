dtmc

// Starting energy -> maximum time for mission
const int e_lim=250;

module rlVehicle
	vehicle_stage: [0..3] init 0;
	// 0-driving, 1-driving in unsafe zone, 2-crashed, 3-goal reached

	u_dist:[0..2] init 0;
	// 0-close (<1/3), 1-medium (<2/3), 2-far (>2/3)

	o_dist:[0..2] init 0;
	// 0-close (<1/3), 1-medium (<2/3), 2-far (>2/3)

	t:[-1..e_lim] init -1;
	// Current time; initialised at -1 so we can probalisitically assign initial state (i.e. at initialisation, what u_dist/o_dist is most likely)

[] t=-1 ->
0.1458:(vehicle_stage'=0)&(u_dist'=2)&(o_dist'=0)&(t'=0) + 
0.2468:(vehicle_stage'=0)&(u_dist'=2)&(o_dist'=2)&(t'=0) + 
0.0232:(vehicle_stage'=0)&(u_dist'=1)&(o_dist'=2)&(t'=0) + 
0.04:(vehicle_stage'=0)&(u_dist'=1)&(o_dist'=1)&(t'=0) + 
0.5272:(vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=0) + 
0.017:(vehicle_stage'=0)&(u_dist'=1)&(o_dist'=0)&(t'=0) ;

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=2 & o_dist=0 -> 
0.9803992066897855: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
0.00843182055415734: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.0050103803017826284: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
0.0017513134851138354: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=0)&(t'=t+1) +
0.004291297943656418: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
4.6392410201691e-05: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
1.159810255042275e-05: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=1)&(t'=t+1) +
4.6392410201691e-05: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
1.159810255042275e-05: (vehicle_stage'=3)&(u_dist'=1)&(o_dist'=0)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=2 & o_dist=1 -> 
0.9812869928141555: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.004583537770711318: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.005151821925210079: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.004695331374875008: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.0019781257181186378: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=1)&(t'=t+1) +
3.7264534721230227e-05: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
9.316133680307557e-06: (vehicle_stage'=3)&(u_dist'=1)&(o_dist'=1)&(t'=t+1) +
0.0021520268801510454: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
9.316133680307557e-06: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=2)&(t'=t+1) +
4.347529050810193e-05: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
2.794840104092267e-05: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
1.8632267360615113e-05: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
6.210755786871705e-06: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=0)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=2 & o_dist=2 -> 
0.9780260318358163: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.0039840882401765785: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.00807269731381458: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.006240508586079055: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.0020043406887308097: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=2)&(t'=t+1) +
0.0015186262273512575: (vehicle_stage'=2)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
3.688970592756091e-05: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
7.377941185512182e-05: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
3.0741421606300764e-05: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=1)&(t'=t+1) +
6.148284321260152e-06: (vehicle_stage'=2)&(u_dist'=1)&(o_dist'=2)&(t'=t+1) +
6.148284321260152e-06: (vehicle_stage'=3)&(u_dist'=1)&(o_dist'=2)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=1 & u_dist=2 & o_dist=2 -> 
0.96143977191732: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.03349964362081254: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.003848895224518888: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.00021382751247327157: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.000819672131147541: (vehicle_stage'=2)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.0001781895937277263: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=1 & o_dist=2 -> 
0.9659998679606523: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=2)&(t'=t+1) +
0.018089390638410245: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.010827226513501024: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=1)&(t'=t+1) +
0.003499042714728989: (vehicle_stage'=3)&(u_dist'=1)&(o_dist'=2)&(t'=t+1) +
0.0009902951079421667: (vehicle_stage'=2)&(u_dist'=1)&(o_dist'=2)&(t'=t+1) +
0.00026407869545124445: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
6.601967386281111e-05: (vehicle_stage'=2)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
6.601967386281111e-05: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=2)&(t'=t+1) +
0.00013203934772562222: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
6.601967386281111e-05: (vehicle_stage'=3)&(u_dist'=1)&(o_dist'=1)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=1 & o_dist=1 -> 
0.9796947693377973: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=1)&(t'=t+1) +
0.012544023653202313: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.0018261663550589156: (vehicle_stage'=3)&(u_dist'=1)&(o_dist'=1)&(t'=t+1) +
0.0032392712726640287: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=0)&(t'=t+1) +
0.00019566068089916952: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=1)&(t'=t+1) +
0.002326188095134571: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=2)&(t'=t+1) +
2.174007565546328e-05: (vehicle_stage'=3)&(u_dist'=1)&(o_dist'=0)&(t'=t+1) +
8.696030262185312e-05: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
4.348015131092656e-05: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
2.174007565546328e-05: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=0)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=1 & u_dist=2 & o_dist=1 -> 
0.9628499557460288: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.031140820794708157: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.0034238598779522057: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.002096240741603391: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
0.0001863325103647459: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=2)&(t'=t+1) +
0.00023291563795593237: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
6.987469138677971e-05: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=0)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=1 & o_dist=0 -> 
0.9841393251438345: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=0)&(t'=t+1) +
0.008604156948115897: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
0.004198413932514383: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=1)&(t'=t+1) +
0.0015549681231534754: (vehicle_stage'=3)&(u_dist'=1)&(o_dist'=0)&(t'=t+1) +
0.0010884776862074327: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=0)&(t'=t+1) +
0.00031099362463069506: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
0.00010366454154356503: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=0)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=0 & o_dist=1 -> 
0.9850467289719627: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=1)&(t'=t+1) +
0.009345794392523364: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=0)&(t'=t+1) +
0.005607476635514018: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=1)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=1 & u_dist=2 & o_dist=0 -> 
0.9700664379061108: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
0.022559684602467694: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
0.0003650434401693802: (vehicle_stage'=3)&(u_dist'=2)&(o_dist'=0)&(t'=t+1) +
0.006935825363218223: (vehicle_stage'=1)&(u_dist'=2)&(o_dist'=1)&(t'=t+1) +
7.300868803387604e-05: (vehicle_stage'=0)&(u_dist'=2)&(o_dist'=1)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=0 & o_dist=0 -> 
0.9862348178137652: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=0)&(t'=t+1) +
0.011336032388663968: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=0)&(t'=t+1) +
0.0024291497975708503: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=1)&(t'=t+1);

[] t>-1 & t<e_lim & vehicle_stage=0 & u_dist=0 & o_dist=2 -> 
0.96: (vehicle_stage'=0)&(u_dist'=0)&(o_dist'=2)&(t'=t+1) +
0.04: (vehicle_stage'=0)&(u_dist'=1)&(o_dist'=2)&(t'=t+1);

endmodule

rewards "unsafe"
	vehicle_stage=1: 1;
endrewards

rewards "close_u"
	u_dist=0: 1;
endrewards

rewards "medium_u"
	u_dist=1: 1;
endrewards

rewards "far_u"
	u_dist=2: 1;
endrewards

rewards "close_o"
	o_dist=0: 1;
endrewards

rewards "medium_o"
	o_dist=1: 1;
endrewards

rewards "far_o"
	o_dist=2: 1;
endrewards