Download Link: https://assignmentchef.com/product/solved-peeking-blackjack-cs221
<br>
<img decoding="async" data-src="blackjack.jpg" class="float-right lazyload" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==">

 <noscript>

  <img decoding="async" class="float-right" src="blackjack.jpg">

 </noscript>

The search algorithms explored in the previous assignment work great when you know exactly the results of your actions. Unfortunately, the real world is not so predictable. One of the key aspects of an effective AI is the ability to reason in the face of uncertainty.

Markov decision processes (MDPs) can be used to formalize uncertain situations. In this homework, you will implement algorithms to find the optimal policy in these situations. You will then formalize a modified version of Blackjack as an MDP, and apply your algorithm to find the optimal policy.

In this problem, you will perform the value iteration updates manually on a very basic game just to solidify your intuitions about solving MDPs. The set of possible states in this game is {-2, -1, 0, 1, 2}. You start at state 0, and if you reach either -2 or 2, the game ends. At each state, you can take one of two actions: {-1, +1}.

If you’re in state <span id="MathJax-Element-1-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-1" class="math"><span id="MathJax-Span-2" class="mrow"><span id="MathJax-Span-3" class="mi">s</span></span></span></span> and choose -1:

<ul>

 <li>You have an 80% chance of reaching the state <span id="MathJax-Element-2-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-4" class="math"><span id="MathJax-Span-5" class="mrow"><span id="MathJax-Span-6" class="mi">s</span><span id="MathJax-Span-7" class="mo">−</span><span id="MathJax-Span-8" class="mn">1</span></span></span></span>.</li>

 <li>You have a 20% chance of reaching the state <span id="MathJax-Element-3-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-9" class="math"><span id="MathJax-Span-10" class="mrow"><span id="MathJax-Span-11" class="mi">s</span><span id="MathJax-Span-12" class="mo">+</span><span id="MathJax-Span-13" class="mn">1</span></span></span></span>.</li>

</ul>

If you’re in state <span id="MathJax-Element-4-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-14" class="math"><span id="MathJax-Span-15" class="mrow"><span id="MathJax-Span-16" class="mi">s</span></span></span></span> and choose +1:

<ul>

 <li>You have a 30% chance of reaching the state <span id="MathJax-Element-5-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-17" class="math"><span id="MathJax-Span-18" class="mrow"><span id="MathJax-Span-19" class="mi">s</span><span id="MathJax-Span-20" class="mo">+</span><span id="MathJax-Span-21" class="mn">1</span></span></span></span>.</li>

 <li>You have a 70% chance of reaching the state <span id="MathJax-Element-6-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-22" class="math"><span id="MathJax-Span-23" class="mrow"><span id="MathJax-Span-24" class="mi">s</span><span id="MathJax-Span-25" class="mo">−</span><span id="MathJax-Span-26" class="mn">1</span></span></span></span>.</li>

</ul>

If your action results in transitioning to state -2, then you receive a reward of 20. If your action results in transitioning to state 2, then your reward is 100. Otherwise, your reward is -5. Assume the discount factor <span id="MathJax-Element-7-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-27" class="math"><span id="MathJax-Span-28" class="mrow"><span id="MathJax-Span-29" class="mi">γ</span></span></span></span> is 1.

<ol class="problem">

 <li id="1a" class="writeup">[3 points] Give the value of <span id="MathJax-Element-8-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-30" class="math"><span id="MathJax-Span-31" class="mrow"><span id="MathJax-Span-32" class="msubsup"><span id="MathJax-Span-33" class="mi">V</span><span id="MathJax-Span-34" class="mtext">opt</span></span><span id="MathJax-Span-35" class="mo">(</span><span id="MathJax-Span-36" class="mi">s</span><span id="MathJax-Span-37" class="mo">)</span></span></span></span> for each state <span id="MathJax-Element-9-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-38" class="math"><span id="MathJax-Span-39" class="mrow"><span id="MathJax-Span-40" class="mi">s</span></span></span></span> after 0, 1, and 2 iterations of value iteration. Iteration 0 just initializes all the values of <span id="MathJax-Element-10-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-41" class="math"><span id="MathJax-Span-42" class="mrow"><span id="MathJax-Span-43" class="mi">V</span></span></span></span> to 0. Terminal states do not have any optimal policies and take on a value of 0.</li>

 <li id="1b" class="writeup">[3 points] What is the resulting optimal policy <span id="MathJax-Element-11-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-44" class="math"><span id="MathJax-Span-45" class="mrow"><span id="MathJax-Span-46" class="msubsup"><span id="MathJax-Span-47" class="mi">π</span><span id="MathJax-Span-48" class="mtext">opt</span></span></span></span></span> for all non-terminal states?</li>

</ol>

Let’s implement value iteration to compute the optimal policy on an arbitrary MDP. Later, we’ll create the specific MDP for Blackjack.

<ol class="problem">

 <li style="list-style-type: none;">

  <ol class="problem">

   <li id="2a" class="code">[3 points] If we add noise to the transitions of an MDP, does the optimal value always get worse? Specifically, consider an MDP with reward function <span id="MathJax-Element-12-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-49" class="math"><span id="MathJax-Span-50" class="mrow"><span id="MathJax-Span-51" class="mtext">Reward</span><span id="MathJax-Span-52" class="mo">(</span><span id="MathJax-Span-53" class="mi">s</span><span id="MathJax-Span-54" class="mo">,</span><span id="MathJax-Span-55" class="mi">a</span><span id="MathJax-Span-56" class="mo">,</span><span id="MathJax-Span-57" class="msup"><span id="MathJax-Span-58" class="mi">s</span><span id="MathJax-Span-59" class="mo">′</span></span><span id="MathJax-Span-60" class="mo">)</span></span></span></span>, states <span id="MathJax-Element-13-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-61" class="math"><span id="MathJax-Span-62" class="mrow"><span id="MathJax-Span-63" class="mtext">States</span></span></span></span>, and transition function <span id="MathJax-Element-14-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-64" class="math"><span id="MathJax-Span-65" class="mrow"><span id="MathJax-Span-66" class="mi">T</span><span id="MathJax-Span-67" class="mo">(</span><span id="MathJax-Span-68" class="mi">s</span><span id="MathJax-Span-69" class="mo">,</span><span id="MathJax-Span-70" class="mi">a</span><span id="MathJax-Span-71" class="mo">,</span><span id="MathJax-Span-72" class="msup"><span id="MathJax-Span-73" class="mi">s</span><span id="MathJax-Span-74" class="mo">′</span></span><span id="MathJax-Span-75" class="mo">)</span></span></span></span>. Let’s define a new MDP which is identical to the original, except that on each action, with probability <span id="MathJax-Element-15-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-76" class="math"><span id="MathJax-Span-77" class="mrow"><span id="MathJax-Span-78" class="mfrac"><span id="MathJax-Span-79" class="mn">1</span><span id="MathJax-Span-80" class="mn">2</span></span></span></span></span>, we randomly jump to one of the states that we could have reached before with positive probability. Formally, this modified transition function is:Let <span id="MathJax-Element-17-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-142" class="math"><span id="MathJax-Span-143" class="mrow"><span id="MathJax-Span-144" class="msubsup"><span id="MathJax-Span-145" class="mi">V</span><span id="MathJax-Span-146" class="mn">1</span></span></span></span></span> be the optimal value function for the original MDP, and <span id="MathJax-Element-18-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-147" class="math"><span id="MathJax-Span-148" class="mrow"><span id="MathJax-Span-149" class="msubsup"><span id="MathJax-Span-150" class="mi">V</span><span id="MathJax-Span-151" class="mn">2</span></span></span></span></span> the optimal value function for the modified MDP. Is it always the case that <span id="MathJax-Element-19-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-152" class="math"><span id="MathJax-Span-153" class="mrow"><span id="MathJax-Span-154" class="msubsup"><span id="MathJax-Span-155" class="mi">V</span><span id="MathJax-Span-156" class="mn">1</span></span><span id="MathJax-Span-157" class="mo">(</span><span id="MathJax-Span-158" class="msubsup"><span id="MathJax-Span-159" class="mi">s</span><span id="MathJax-Span-160" class="mtext">start</span></span><span id="MathJax-Span-161" class="mo">)</span><span id="MathJax-Span-162" class="mo">≥</span><span id="MathJax-Span-163" class="msubsup"><span id="MathJax-Span-164" class="mi">V</span><span id="MathJax-Span-165" class="mn">2</span></span><span id="MathJax-Span-166" class="mo">(</span><span id="MathJax-Span-167" class="msubsup"><span id="MathJax-Span-168" class="mi">s</span><span id="MathJax-Span-169" class="mtext">start</span></span><span id="MathJax-Span-170" class="mo">)</span></span></span></span>? If so, prove it in <code>blackjack.pdf</code> and put <code>return None</code> for each of the code blocks. Otherwise, construct a counterexample by filling out <code>CounterexampleMDP</code> in <code><a href="submission.py">submission.py</a></code>.</li>

   <li id="2b" class="writeup">[3 points] Suppose we have an acyclic MDP for which we want to find the optimal value at each node. We could run value iteration, which would require multiple iterations — but it would be nice to be more efficient for MDPs with this acyclic property. Briefly explain an algorithm that will allow us to compute <span id="MathJax-Element-20-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-171" class="math"><span id="MathJax-Span-172" class="mrow"><span id="MathJax-Span-173" class="msubsup"><span id="MathJax-Span-174" class="mi">V</span><span id="MathJax-Span-175" class="mtext">opt</span></span></span></span></span> for each node with only a single pass over all the <span id="MathJax-Element-21-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-176" class="math"><span id="MathJax-Span-177" class="mrow"><span id="MathJax-Span-178" class="mo">(</span><span id="MathJax-Span-179" class="mi">s</span><span id="MathJax-Span-180" class="mo">,</span><span id="MathJax-Span-181" class="mi">a</span><span id="MathJax-Span-182" class="mo">,</span><span id="MathJax-Span-183" class="msup"><span id="MathJax-Span-184" class="mi">s</span><span id="MathJax-Span-185" class="mo">′</span></span><span id="MathJax-Span-186" class="mo">)</span></span></span></span> triples.</li>

   <li id="2c" class="writeup">[3 points] Suppose we have an MDP with states <span id="MathJax-Element-22-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-187" class="math"><span id="MathJax-Span-188" class="mrow"><span id="MathJax-Span-189" class="mtext">States</span></span></span></span> a discount factor <span id="MathJax-Element-23-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-190" class="math"><span id="MathJax-Span-191" class="mrow"><span id="MathJax-Span-192" class="mi">γ</span><span id="MathJax-Span-193" class="mo">&lt;</span><span id="MathJax-Span-194" class="mn">1</span></span></span></span>, but we have an MDP solver that only can solve MDPs with discount <span id="MathJax-Element-24-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-195" class="math"><span id="MathJax-Span-196" class="mrow"><span id="MathJax-Span-197" class="mn">1</span></span></span></span>. How can leverage the MDP solver to solve the original MDP?Let us define a new MDP with states <span id="MathJax-Element-25-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-198" class="math"><span id="MathJax-Span-199" class="mrow"><span id="MathJax-Span-200" class="msup"><span id="MathJax-Span-201" class="mtext">States</span><span id="MathJax-Span-202" class="mo">′</span></span><span id="MathJax-Span-203" class="mo">=</span><span id="MathJax-Span-204" class="mtext">States</span><span id="MathJax-Span-205" class="mo">∪</span><span id="MathJax-Span-206" class="mo">{</span><span id="MathJax-Span-207" class="mi">o</span><span id="MathJax-Span-208" class="mo">}</span></span></span></span>, where <span id="MathJax-Element-26-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-209" class="math"><span id="MathJax-Span-210" class="mrow"><span id="MathJax-Span-211" class="mi">o</span></span></span></span> is a new state. Let’s use the same actions (<span id="MathJax-Element-27-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-212" class="math"><span id="MathJax-Span-213" class="mrow"><span id="MathJax-Span-214" class="msup"><span id="MathJax-Span-215" class="mtext">Actions</span><span id="MathJax-Span-216" class="mo">′</span></span><span id="MathJax-Span-217" class="mo">(</span><span id="MathJax-Span-218" class="mi">s</span><span id="MathJax-Span-219" class="mo">)</span><span id="MathJax-Span-220" class="mo">=</span><span id="MathJax-Span-221" class="mtext">Actions</span><span id="MathJax-Span-222" class="mo">(</span><span id="MathJax-Span-223" class="mi">s</span><span id="MathJax-Span-224" class="mo">)</span></span></span></span>), but we need to keep the discount <span id="MathJax-Element-28-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-225" class="math"><span id="MathJax-Span-226" class="mrow"><span id="MathJax-Span-227" class="msup"><span id="MathJax-Span-228" class="mi">γ</span><span id="MathJax-Span-229" class="mo">′</span></span><span id="MathJax-Span-230" class="mo">=</span><span id="MathJax-Span-231" class="mn">1</span></span></span></span>. Your job is to define new transition probabilities <span id="MathJax-Element-29-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-232" class="math"><span id="MathJax-Span-233" class="mrow"><span id="MathJax-Span-234" class="msup"><span id="MathJax-Span-235" class="mi">T</span><span id="MathJax-Span-236" class="mo">′</span></span><span id="MathJax-Span-237" class="mo">(</span><span id="MathJax-Span-238" class="mi">s</span><span id="MathJax-Span-239" class="mo">,</span><span id="MathJax-Span-240" class="mi">a</span><span id="MathJax-Span-241" class="mo">,</span><span id="MathJax-Span-242" class="msup"><span id="MathJax-Span-243" class="mi">s</span><span id="MathJax-Span-244" class="mo">′</span></span><span id="MathJax-Span-245" class="mo">)</span></span></span></span> and rewards <span id="MathJax-Element-30-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-246" class="math"><span id="MathJax-Span-247" class="mrow"><span id="MathJax-Span-248" class="msup"><span id="MathJax-Span-249" class="mtext">Reward</span><span id="MathJax-Span-250" class="mo">′</span></span><span id="MathJax-Span-251" class="mo">(</span><span id="MathJax-Span-252" class="mi">s</span><span id="MathJax-Span-253" class="mo">,</span><span id="MathJax-Span-254" class="mi">a</span><span id="MathJax-Span-255" class="mo">,</span><span id="MathJax-Span-256" class="msup"><span id="MathJax-Span-257" class="mi">s</span><span id="MathJax-Span-258" class="mo">′</span></span><span id="MathJax-Span-259" class="mo">)</span></span></span></span> in terms of the old MDP such that the optimal values <span id="MathJax-Element-31-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-260" class="math"><span id="MathJax-Span-261" class="mrow"><span id="MathJax-Span-262" class="msubsup"><span id="MathJax-Span-263" class="mi">V</span><span id="MathJax-Span-264" class="mtext">opt</span></span><span id="MathJax-Span-265" class="mo">(</span><span id="MathJax-Span-266" class="mi">s</span><span id="MathJax-Span-267" class="mo">)</span></span></span></span> for all <span id="MathJax-Element-32-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-268" class="math"><span id="MathJax-Span-269" class="mrow"><span id="MathJax-Span-270" class="mi">s</span><span id="MathJax-Span-271" class="mo">∈</span><span id="MathJax-Span-272" class="mtext">States</span></span></span></span> are the equal under the original MDP and the new MDP.Hint: If you’re not sure how to approach this problem, go back to Percy’s notes from the first MDP lecture and read closely the slides on convergence, toward the end of the deck.</li>

  </ol></li>

</ol>

Now that we gotten a bit of practice with general-purpose MDP algorithms, let’s use them to play (a modified version of) Blackjack. For this problem, you will be creating an MDP to describe states, actions, and rewards in this game.

For our version of Blackjack, the deck can contain an arbitrary collection of cards with different face values. At the start of the game, the deck contains the same number of each cards of each face value; we call this number the ‘multiplicity’. For example, a standard deck of 52 cards would have face values <span id="MathJax-Element-33-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-273" class="math"><span id="MathJax-Span-274" class="mrow"><span id="MathJax-Span-275" class="mo">[</span><span id="MathJax-Span-276" class="mn">1</span><span id="MathJax-Span-277" class="mo">,</span><span id="MathJax-Span-278" class="mn">2</span><span id="MathJax-Span-279" class="mo">,</span><span id="MathJax-Span-280" class="mo">…</span><span id="MathJax-Span-281" class="mo">,</span><span id="MathJax-Span-282" class="mn">13</span><span id="MathJax-Span-283" class="mo">]</span></span></span></span> and multiplicity 4. You could also have a deck with face values <span id="MathJax-Element-34-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-284" class="math"><span id="MathJax-Span-285" class="mrow"><span id="MathJax-Span-286" class="mo">[</span><span id="MathJax-Span-287" class="mn">1</span><span id="MathJax-Span-288" class="mo">,</span><span id="MathJax-Span-289" class="mn">5</span><span id="MathJax-Span-290" class="mo">,</span><span id="MathJax-Span-291" class="mn">20</span><span id="MathJax-Span-292" class="mo">]</span></span></span></span>; if we used multiplicity 10 in this case, there would be 30 cards in total (10 each of 1s, 5s, and 20s). The deck is shuffled, meaning that each permutation of the cards is equally likely.

The game occurs in a sequence of rounds. Each round, the player either (i) takes the next card from the top of the deck (costing nothing), (ii) peeks at the top card (costing <code>peekCost</code>, in which case the next round, that card will be drawn), or (iii) quits the game. (Note: it is not possible to peek twice in a row; if the player peeks twice in a row, then <code>succAndProbReward()</code> should return <code>[]</code>.)

The game continues until one of the following conditions becomes true:

<ol class="problem">

 <li style="list-style-type: none;">

  <ol class="problem">

   <li style="list-style-type: none;">

    <ul>

     <li>The player quits, in which case her reward is the sum of the face values of the cards in her hand.</li>

     <li>The player takes a card and “goes bust”. This means that the sum of the face values of the cards in her hand is strictly greater than the threshold specified at the start of the game. If this happens, her reward is 0.</li>

     <li>The deck runs out of cards, in which case it is as if she quits, and she gets a reward which is the sum of the cards in her hand.</li>

    </ul></li>

  </ol></li>

</ol>

In this problem, your state <span id="MathJax-Element-35-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-293" class="math"><span id="MathJax-Span-294" class="mrow"><span id="MathJax-Span-295" class="mi">s</span></span></span></span> will be represented as a 3-element tuple:

<blockquote>

 <code>(totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts)</code>

</blockquote>

<ol class="problem">

 <li style="list-style-type: none;">

  <ol class="problem">

   As an example, assume the deck has card values

  </ol></li>

</ol>

<span id="MathJax-Element-36-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-296" class="math"><span id="MathJax-Span-297" class="mrow"><span id="MathJax-Span-298" class="mo">[</span><span id="MathJax-Span-299" class="mn">1</span><span id="MathJax-Span-300" class="mo">,</span><span id="MathJax-Span-301" class="mn">2</span><span id="MathJax-Span-302" class="mo">,</span><span id="MathJax-Span-303" class="mn">3</span><span id="MathJax-Span-304" class="mo">]</span></span></span></span>

<ol class="problem">

 <li style="list-style-type: none;">

  <ol class="problem">

    with multiplicity 1, and the threshold is 4. Initially, the player has no cards, so her total is 0; this corresponds to state

  </ol></li>

</ol>

<code>(0, None, (1, 1, 1))</code>

<ol class="problem">

 <li style="list-style-type: none;">

  <ol class="problem">

   <li style="list-style-type: none;">

    <ol class="problem">

     . At this point, she can take, peek, or quit.




     <li style="list-style-type: none;">

      <ul>

       <li>If she takes, the three possible successor states (each of which has equal probability of <span id="MathJax-Element-37-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-305" class="math"><span id="MathJax-Span-306" class="mrow"><span id="MathJax-Span-307" class="mn">1</span><span id="MathJax-Span-308" class="texatom"><span id="MathJax-Span-309" class="mrow"><span id="MathJax-Span-310" class="mo">/</span></span></span><span id="MathJax-Span-311" class="mn">3</span></span></span></span>) are:

        <blockquote>

         <code>(1, None, (0, 1, 1))</code><code>(2, None, (1, 0, 1))</code><code>(3, None, (1, 1, 0))</code>

        </blockquote>She will receive a reward of 0 for reaching any of these states. (Remember, even though she now has a card in her hand for which she may receive a reward at the end of the game, the reward is not actually granted until the game ends.)</li>

       <li>If she peeks, the three possible successor states are:

        <blockquote>

         <code>(0, 0, (1, 1, 1))</code><code>(0, 1, (1, 1, 1))</code><code>(0, 2, (1, 1, 1))</code>

        </blockquote>She will receive (immediate) reward <code>-peekCost</code> for reaching any of these states. Things to remember about the states after a peek action:

        <ul>

         <li>From <code>(0, 0, (1, 1, 1))</code>, taking a card will lead to the state <code>(1, None, (0, 1, 1))</code> deterministically.</li>

         <li>The second element of the state tuple is not the face value of the card that will be drawn next, but the index into the deck (the third element of the state tuple) of the card that will be drawn next. In other words, the second element will always be between 0 and <code>len(deckCardCounts)-1</code>, inclusive.</li>

        </ul></li>

       <li>If she quits, then the resulting state will be <code>(0, None, None)</code>. (Remember that setting the deck to <code>None</code> signifies the end of the game.)</li>

      </ul></li>

    </ol></li>

  </ol>As another example, let’s say the player’s current state is</li>

</ol>

<code>(3, None, (1, 1, 0))</code>

<ol class="problem">

 <li style="list-style-type: none;">

  <ol class="problem">

   , and the threshold remains 4.




   <li style="list-style-type: none;">

    <ul>

     <li>If she quits, the successor state will be <code>(3, None, None)</code>.</li>

     <li>If she takes, the successor states are <code>(3 + 1, None, (0, 1, 0))</code> or <code>(3 + 2, None, None)</code>. Note that in the second successor state, the deck is set to <code>None</code> to signify the game ended with a bust. You should also set the deck to <code>None</code> if the deck runs out of cards.</li>

    </ul></li>

   <li style="list-style-type: none;">

    <ol class="problem">

     <li id="3a" class="code">[10 points] Implement the game of Blackjack as an MDP by filling out the <code>succAndProbReward()</code> function of class <code>BlackjackMDP</code>.</li>

     <li id="3b" class="code">[4 points] Let’s say you’re running a casino, and you’re trying to design a deck to make people peek a lot. Assuming a fixed threshold of 20, and a peek cost of 1, design a deck where for at least 10% of states, the optimal policy is to peek. Fill out the function <code>peekingMDP()</code> to return an instance of <code>BlackjackMDP</code> where the optimal action is to peek in at least 10% of states.</li>

    </ol></li>

  </ol></li>

</ol>

So far, we’ve seen how MDP algorithms can take an MDP which describes the full dynamics of the game and return an optimal policy. But suppose you go into a casino, and no one tells you the rewards or the transitions. We will see how reinforcement learning can allow you to play the game and learn its rules &amp; strategy at the same time!

<ol class="problem">

 <li style="list-style-type: none;">

  <ol class="problem">

   <li id="4a" class="code">[8 points] You will first implement a generic Q-learning algorithm <code>QLearningAlgorithm</code>, which is an instance of an <code>RLAlgorithm</code>. As discussed in class, reinforcement learning algorithms are capable of executing a policy while simultaneously improving that policy. Look in <code>simulate()</code>, in <code><a href="util.py">util.py</a></code> to see how the <code>RLAlgorithm</code> will be used. In short, your <code>QLearningAlgorithm</code> will be run in a simulation of the MDP, and will alternately be asked for an action to perform in a given state (<code>QLearningAlgorithm.getAction</code>), and then be informed of the result of that action (<code>QLearningAlgorithm.incorporateFeedback</code>), so that it may learn better actions to perform in the future.We are using Q-learning with function approximation, which means <span id="MathJax-Element-38-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-312" class="math"><span id="MathJax-Span-313" class="mrow"><span id="MathJax-Span-314" class="msubsup"><span id="MathJax-Span-315" class="texatom"><span id="MathJax-Span-316" class="mrow"><span id="MathJax-Span-317" class="munderover"><span id="MathJax-Span-318" class="mi">Q</span><span id="MathJax-Span-319" class="mo">^</span></span></span></span><span id="MathJax-Span-320" class="mtext">opt</span></span><span id="MathJax-Span-321" class="mo">(</span><span id="MathJax-Span-322" class="mi">s</span><span id="MathJax-Span-323" class="mo">,</span><span id="MathJax-Span-324" class="mi">a</span><span id="MathJax-Span-325" class="mo">)</span><span id="MathJax-Span-326" class="mo">=</span><span id="MathJax-Span-327" class="texatom"><span id="MathJax-Span-328" class="mrow"><span id="MathJax-Span-329" class="mi">w</span></span></span><span id="MathJax-Span-330" class="mo">⋅</span><span id="MathJax-Span-331" class="mi">ϕ</span><span id="MathJax-Span-332" class="mo">(</span><span id="MathJax-Span-333" class="mi">s</span><span id="MathJax-Span-334" class="mo">,</span><span id="MathJax-Span-335" class="mi">a</span><span id="MathJax-Span-336" class="mo">)</span></span></span></span>, where in code, <span id="MathJax-Element-39-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-337" class="math"><span id="MathJax-Span-338" class="mrow"><span id="MathJax-Span-339" class="texatom"><span id="MathJax-Span-340" class="mrow"><span id="MathJax-Span-341" class="mi">w</span></span></span></span></span></span> is <code>self.weights</code>, <span id="MathJax-Element-40-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-342" class="math"><span id="MathJax-Span-343" class="mrow"><span id="MathJax-Span-344" class="mi">ϕ</span></span></span></span> is the <code>featureExtractor</code> function, and <span id="MathJax-Element-41-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-345" class="math"><span id="MathJax-Span-346" class="mrow"><span id="MathJax-Span-347" class="msubsup"><span id="MathJax-Span-348" class="texatom"><span id="MathJax-Span-349" class="mrow"><span id="MathJax-Span-350" class="munderover"><span id="MathJax-Span-351" class="mi">Q</span><span id="MathJax-Span-352" class="mo">^</span></span></span></span><span id="MathJax-Span-353" class="mtext">opt</span></span></span></span></span> is <code>self.getQ</code>.We have implemented <code>QLearningAlgorithm.getAction</code> as a simple <span id="MathJax-Element-42-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-354" class="math"><span id="MathJax-Span-355" class="mrow"><span id="MathJax-Span-356" class="mi">ϵ</span></span></span></span>-greedy policy. Your job is to implement <code>QLearningAlgorithm.incorporateFeedback()</code>, which should take an <span id="MathJax-Element-43-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-357" class="math"><span id="MathJax-Span-358" class="mrow"><span id="MathJax-Span-359" class="mo">(</span><span id="MathJax-Span-360" class="mi">s</span><span id="MathJax-Span-361" class="mo">,</span><span id="MathJax-Span-362" class="mi">a</span><span id="MathJax-Span-363" class="mo">,</span><span id="MathJax-Span-364" class="mi">r</span><span id="MathJax-Span-365" class="mo">,</span><span id="MathJax-Span-366" class="msup"><span id="MathJax-Span-367" class="mi">s</span><span id="MathJax-Span-368" class="mo">′</span></span><span id="MathJax-Span-369" class="mo">)</span></span></span></span> tuple and update <code>self.weights</code> according to the standard Q-learning update.</li>

   <li id="4b" class="writeup">[4 points] Now let’s apply Q-learning to an MDP and see how well it performs in comparison with value iteration. First, call <code>simulate</code> using your Q-learning code and the <code>identityFeatureExtractor()</code> on the MDP <code>smallMDP</code> (defined for you in <code><a href="submission.py">submission.py</a></code>), with 30000 trials. How does the Q-learning policy compare with a policy learned by value iteration (i.e., for how many states do they produce a different action)? (Don’t forget to set the explorationProb of your Q-learning algorithm to 0 after learning the policy.) Now run <code>simulate()</code> on <code>largeMDP</code>, again with 30000 trials. How does the policy learned in this case compare to the policy learned by value iteration? What went wrong?</li>

   <li id="4c" class="code">[5 points] To address the problems explored in the previous exercise, let’s incorporate some domain knowledge to improve generalization. This way, the algorithm can use what it has learned about some states to improve its prediction performance on other states. Implement <code>blackjackFeatureExtractor</code> as described in the code comments. Using this feature extractor, you should be able to get pretty close to the optimum on the <code>largeMDP</code>.</li>

   <li id="4d" class="writeup">[4 points] Sometimes, we might reasonably wonder how an optimal policy learned for one MDP might perform if applied to another MDP with similar structure but slightly different characteristics. For example, imagine that you created an MDP to choose an optimal strategy for playing “traditional” blackjack, with a standard card deck and a threshold of 21. You’re living it up in Vegas every weekend, but the casinos get wise to your approach and decide to make a change to the game to disrupt your strategy: going forward, the threshold for the blackjack tables is 17 instead of 21. If you continued playing the modified game with your original policy, how well would you do? (This is just a hypothetical example; we won’t look specifically at the blackjack game in this problem.)To explore this scenario, let’s take a brief look at how a policy learned using value iteration responds to a change in the rules of the MDP.

    <ul>

     <li>First, run value iteration on the <code>originalMDP</code> (defined for you in <code><a href="submission.py">submission.py</a></code>) to compute an optimal policy for that MDP.</li>

     <li>Next, simulate your policy on <code>newThresholdMDP</code> (also defined for you in <code><a href="submission.py">submission.py</a></code>) by calling <code>simulate</code> with an instance of <code>FixedRLAlgorithm</code> that has been instantiated using the policy you computed with value iteration. What is the expected reward from this simulation? Hint: read the documentation (comments) for the <code>simulate</code> function in util.py, and look specifically at the format of the function’s return value.</li>

     <li>Now try simulating Q-learning directly on <code>newThresholdMDP</code> instead. What is your expected reward under the new Q-learning policy? Provide some explanation for how the rewards compare, and why they are different.</li>

    </ul></li>

  </ol></li>

</ol>

5/5 - (2 votes)

This (and every) assignment has a written part and a programming part.

The full assignment with our supporting code and scripts can be downloaded as <a href="../blackjack.zip">blackjack.zip</a>.

<ol class="problem">

 <li class="writeup template">This icon means a written answer is expected in <code>blackjack.pdf</code>.</li>

 <li class="code template">This icon means you should write code in <code><a href="submission.py">submission.py</a></code>.</li>

</ol>

You should modify the code in <code><a href="submission.py">submission.py</a></code> between

<pre># BEGIN_YOUR_CODE</pre>

and

<pre># END_YOUR_CODE</pre>

but you can add other helper functions outside this block if you want. Do not make changes to files other than <code><a href="submission.py">submission.py</a></code>.Your code will be evaluated on two types of test cases, <b>basic</b> and <b>hidden</b>, which you can see in <code><a href="grader.py">grader.py</a></code>. Basic tests, which are fully provided to you, do not stress your code with large inputs or tricky corner cases. Hidden tests are more complex and do stress your code. The inputs of hidden tests are provided in <code><a href="grader.py">grader.py</a></code>, but the correct outputs are not. To run the tests, you will need to have <code><a href="graderUtil.py">graderUtil.py</a></code> in the same directory as your code and <code><a href="grader.py">grader.py</a></code>. Then, you can run all the tests by typing

<pre>python grader.py</pre>

This will tell you only whether you passed the basic tests. On the hidden tests, the script will alert you if your code takes too long or crashes, but does not say whether you got the correct output. You can also run a single test (e.g., <code>3a-0-basic</code>) by typing

<pre>python grader.py 3a-0-basic</pre>

We strongly encourage you to read and understand the test cases, create your own test cases, and not just blindly run <code><a href="grader.py">grader.py</a></code>.

<hr>

<b> </b>General Instructions

Problem 1: Value Iteration

Problem 2: Transforming MDPs

<span id="MathJax-Element-16-Frame" class="MathJax" tabindex="0"><span id="MathJax-Span-81" class="math"><span id="MathJax-Span-82" class="mrow"><span id="MathJax-Span-83" class="msup"><span id="MathJax-Span-84" class="mi">T</span><span id="MathJax-Span-85" class="mo">′</span></span><span id="MathJax-Span-86" class="mo">(</span><span id="MathJax-Span-87" class="mi">s</span><span id="MathJax-Span-88" class="mo">,</span><span id="MathJax-Span-89" class="mi">a</span><span id="MathJax-Span-90" class="mo">,</span><span id="MathJax-Span-91" class="msup"><span id="MathJax-Span-92" class="mi">s</span><span id="MathJax-Span-93" class="mo">′</span></span><span id="MathJax-Span-94" class="mo">)</span><span id="MathJax-Span-95" class="mo">=</span><span id="MathJax-Span-96" class="mfrac"><span id="MathJax-Span-97" class="mn">1</span><span id="MathJax-Span-98" class="mn">2</span></span><span id="MathJax-Span-99" class="mi">T</span><span id="MathJax-Span-100" class="mo">(</span><span id="MathJax-Span-101" class="mi">s</span><span id="MathJax-Span-102" class="mo">,</span><span id="MathJax-Span-103" class="mi">a</span><span id="MathJax-Span-104" class="mo">,</span><span id="MathJax-Span-105" class="msup"><span id="MathJax-Span-106" class="mi">s</span><span id="MathJax-Span-107" class="mo">′</span></span><span id="MathJax-Span-108" class="mo">)</span><span id="MathJax-Span-109" class="mo">+</span><span id="MathJax-Span-110" class="mfrac"><span id="MathJax-Span-111" class="mn">1</span><span id="MathJax-Span-112" class="mn">2</span></span><span id="MathJax-Span-113" class="mo">⋅</span><span id="MathJax-Span-114" class="mfrac"><span id="MathJax-Span-115" class="mn">1</span><span id="MathJax-Span-116" class="mrow"><span id="MathJax-Span-117" class="texatom"><span id="MathJax-Span-118" class="mrow"><span id="MathJax-Span-119" class="mo">|</span></span></span><span id="MathJax-Span-120" class="mo">{</span><span id="MathJax-Span-121" class="msup"><span id="MathJax-Span-122" class="mi">s</span><span id="MathJax-Span-123" class="mo">′′</span></span><span id="MathJax-Span-124" class="mo">:</span><span id="MathJax-Span-125" class="mi">T</span><span id="MathJax-Span-126" class="mo">(</span><span id="MathJax-Span-127" class="mi">s</span><span id="MathJax-Span-128" class="mo">,</span><span id="MathJax-Span-129" class="mi">a</span><span id="MathJax-Span-130" class="mo">,</span><span id="MathJax-Span-131" class="msup"><span id="MathJax-Span-132" class="mi">s</span><span id="MathJax-Span-133" class="mo">′′</span></span><span id="MathJax-Span-134" class="mo">)</span><span id="MathJax-Span-135" class="mo">&gt;</span><span id="MathJax-Span-136" class="mn">0</span><span id="MathJax-Span-137" class="mo">}</span><span id="MathJax-Span-138" class="texatom"><span id="MathJax-Span-139" class="mrow"><span id="MathJax-Span-140" class="mo">|</span></span></span></span></span><span id="MathJax-Span-141" class="mo">.</span></span></span></span>

Problem 3: Peeking Blackjack