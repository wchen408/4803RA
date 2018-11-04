### Goal

Investigate whether exploiting users' digital finger printings  (geographic, browser, OS, device) helps Recommendation Agent (RA) make better recommendations

### Assumption

* The website sells $N$ products and displays a subset $S \subsetneq N$ each time. $|S| = 10$ 
* Each user come in with the intention to purchase a product that satisfy a set of critieria $C$. Each user proxy will be drew from the census data, with his critieria set $C$ generated from according to his income level. 
* Website sells product(s) that satisfy $C$

### Procedure

Two recommendation engines $RA_1$, $RA_2$. Where $RA_2$ takes in extra degree of information in generating subset $S$ and $RA_1$ does not. 

$t = 0$: 

​	User apply intial filter

For $t = 1...j$ where $\nexists \mbox{ Product P} \in S_{t} \mbox{ s.t. } P \mbox{ satisfies } C $: 

​	User clicks on an item $P_{t}\prime$ most closetly resembles $P$, the item he is looking for. RA readjust weights of all product based on this click pattern and regenerate subset $S_{t+1}$

At $t = j​$: 

​	$\exists \mbox{ Product P} \in S_{t} \mbox{ s.t. } P \mbox{ satisfies } C$

Compare converge time of $RA_1$ $RA_2$

 