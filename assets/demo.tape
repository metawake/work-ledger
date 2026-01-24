# Work Ledger Demo - Catch an Agent Bug
#
# Generate: vhs assets/demo.tape
# Output: assets/demo.gif

Output assets/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 800
Set Height 500
Set Padding 30
Set TypingSpeed 40ms
Set Theme "Catppuccin Mocha"

# SCENE 1: THE PROBLEM
Hide
Type "bash assets/demo_scene.sh 1"
Enter
Show
Sleep 4s

# SCENE 2: THE INTEGRATION
Hide
Type "bash assets/demo_scene.sh 2"
Enter
Show
Sleep 5s

# SCENE 3: THE PAYOFF
Hide
Type "bash assets/demo_scene.sh 3a"
Enter
Show
Sleep 2s

Hide
Type "bash assets/demo_scene.sh 3b"
Enter
Show
Sleep 8s

# SCENE 4: CTA
Hide
Type "bash assets/demo_scene.sh 4"
Enter
Show
Sleep 3s
