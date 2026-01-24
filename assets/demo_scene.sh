#!/bin/bash
# Usage: ./demo_scene.sh <scene_number>

case $1 in
  1)
    clear
    echo ""
    echo -e "\033[1;36m  THE PROBLEM\033[0m"
    echo ""
    echo "  Your support agent started issuing refunds."
    echo "  You changed one word in the prompt."
    echo "  What happened?"
    echo ""
    ;;
  2)
    clear
    echo ""
    echo -e "\033[1;36m  THE FIX: 3 LINES\033[0m"
    echo ""
    cat samples/demo_snippet.py
    ;;
  3a)
    clear
    echo ""
    echo -e "\033[1;36m  FIND THE BUG\033[0m"
    echo ""
    echo "  Compare yesterday's run vs today's run:"
    echo ""
    echo -e "  \033[0;90m$ work-ledger diff ./runs yesterday today\033[0m"
    echo ""
    ;;
  3b)
    echo "  Comparing runs:"
    echo -e "    Expected: \033[0;37msupport-agent-jan-30\033[0m (before prompt change)"
    echo -e "    Actual:   \033[0;37msupport-agent-jan-31\033[0m (after prompt change)"
    echo ""
    echo -e "  Similarity: \033[1;33m71%\033[0m"
    echo ""
    echo "  Step changes:"
    echo -e "    \033[1;31m+ refund_payment [tool]   <- BUG! Not in baseline.\033[0m"
    echo ""
    echo -e "  \033[1;32mFound it. The new prompt triggers unauthorized refunds.\033[0m"
    echo ""
    ;;
  4)
    clear
    echo ""
    echo -e "\033[1;36m  GET STARTED\033[0m"
    echo ""
    echo "  pip install work-ledger"
    echo ""
    echo -e "  \033[0;34mgithub.com/metawake/workledger\033[0m"
    echo ""
    ;;
esac
