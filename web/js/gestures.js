/**
 * ASL Gestures & Symbol Showcase Library
 * Contains definitions for:
 * - 26 Alphabet Signs (A to Z)
 * - 10 Number Signs (0 to 9)
 * - 14 Common Signs & Everyday Phrases (Hello, Thank You, Yes, No, Please, I Love You, Help, Sorry, Good, Bad, Water, More, Peace, OK)
 * - Rule-based gesture recognition heuristics for natural expressions
 */

const ASL_DICTIONARY = {
    alphabet: [
        { letter: 'A', name: 'Letter A', category: 'alphabet', description: 'Make a fist with the thumb resting alongside the index finger.', tips: 'Thumb rests upright against the side of the fist, not tucked inside.', commonWords: ['Apple', 'About', 'Always'] },
        { letter: 'B', name: 'Letter B', category: 'alphabet', description: 'Hold four fingers straight up together with the thumb folded across the palm.', tips: 'Fingers should be held tight together, thumb resting over palm.', commonWords: ['Book', 'Baby', 'Beautiful'] },
        { letter: 'C', name: 'Letter C', category: 'alphabet', description: 'Curved hand forming the letter "C" shape with fingers and thumb.', tips: 'Fingers curve together like holding a cup.', commonWords: ['Cat', 'Care', 'Coffee'] },
        { letter: 'D', name: 'Letter D', category: 'alphabet', description: 'Index finger points straight up; middle, ring, little fingers curve to touch the thumb tip forming an "O".', tips: 'Only the index finger stands straight up.', commonWords: ['Dog', 'Day', 'Dream'] },
        { letter: 'E', name: 'Letter E', category: 'alphabet', description: 'All fingers curl down with fingertips resting lightly on the thumb folded below them.', tips: 'Fingernails face forward resting above the thumb.', commonWords: ['Eat', 'Eye', 'Earth'] },
        { letter: 'F', name: 'Letter F', category: 'alphabet', description: 'Index finger and thumb touch at tips forming a circle ("OK" shape), with middle, ring, and pinky standing tall.', tips: 'Three outer fingers are spread apart and straight.', commonWords: ['Friend', 'Family', 'Food'] },
        { letter: 'G', name: 'Letter G', category: 'alphabet', description: 'Index finger and thumb point horizontally parallel, other fingers closed in fist.', tips: 'Think of pinching something small sideways.', commonWords: ['Good', 'Give', 'Great'] },
        { letter: 'H', name: 'Letter H', category: 'alphabet', description: 'Index and middle fingers extend horizontally together, thumb tucked over ring finger.', tips: 'Like G, but with both index and middle fingers pointing sideways.', commonWords: ['Hello', 'Home', 'Help'] },
        { letter: 'I', name: 'Letter I', category: 'alphabet', description: 'Make a fist and extend only the pinky finger straight up.', tips: 'Thumb rests over the folded fingers; only little finger is upright.', commonWords: ['Ice', 'Idea', 'Important'] },
        { letter: 'J', name: 'Letter J', category: 'alphabet', description: 'Start with pinky up (like I) and trace the letter "J" hook in the air.', tips: 'Curve downward and scoop upward with pinky.', commonWords: ['Joy', 'Job', 'Journey'] },
        { letter: 'K', name: 'Letter K', category: 'alphabet', description: 'Index finger straight up, middle finger angled forward, thumb placed between them at the knuckle.', tips: 'Forms a "K" profile sideways.', commonWords: ['Kind', 'Know', 'Keep'] },
        { letter: 'L', name: 'Letter L', category: 'alphabet', description: 'Index finger straight up and thumb extended perpendicular forming an "L" shape.', tips: 'Other three fingers tucked into palm.', commonWords: ['Love', 'Learn', 'Life'] },
        { letter: 'M', name: 'Letter M', category: 'alphabet', description: 'Make a fist with the thumb tucked under the first three fingers (index, middle, ring).', tips: 'Three finger bumps visible over the thumb.', commonWords: ['Mother', 'Morning', 'Music'] },
        { letter: 'N', name: 'Letter N', category: 'alphabet', description: 'Make a fist with the thumb tucked under the first two fingers (index, middle).', tips: 'Two finger bumps visible over the thumb.', commonWords: ['Name', 'Night', 'Nature'] },
        { letter: 'O', name: 'Letter O', category: 'alphabet', description: 'All fingers curl to meet thumb tip forming an "O" circular tube.', tips: 'Clean circular silhouette.', commonWords: ['Open', 'Order', 'Opportunity'] },
        { letter: 'P', name: 'Letter P', category: 'alphabet', description: 'Like "K" but pointing downwards (index straight down, middle angled forward).', tips: 'Thumb rests at joint between index and middle.', commonWords: ['People', 'Peace', 'Please'] },
        { letter: 'Q', name: 'Letter Q', category: 'alphabet', description: 'Like "G" but pointing downwards toward the floor.', tips: 'Index and thumb point down like pinching downward.', commonWords: ['Quick', 'Quiet', 'Question'] },
        { letter: 'R', name: 'Letter R', category: 'alphabet', description: 'Cross the middle finger over the index finger like making a "good luck" wish.', tips: 'Thumb secures ring and pinky fingers against palm.', commonWords: ['Right', 'Read', 'Ready'] },
        { letter: 'S', name: 'Letter S', category: 'alphabet', description: 'Make a tight fist with thumb wrapped across the front of all fingers.', tips: 'Thumb sits horizontally across the fingers (unlike A where thumb is at side).', commonWords: ['Smile', 'School', 'Sun'] },
        { letter: 'T', name: 'Letter T', category: 'alphabet', description: 'Make a fist with thumb tucked under the index finger only.', tips: 'Single finger bump visible over the thumb.', commonWords: ['Time', 'Thank', 'Today'] },
        { letter: 'U', name: 'Letter U', category: 'alphabet', description: 'Index and middle fingers stand straight up pressed tightly together.', tips: 'Fingers together and upright; thumb over ring finger.', commonWords: ['Understand', 'Use', 'Unique'] },
        { letter: 'V', name: 'Letter V', category: 'alphabet', description: 'Index and middle fingers extend upright spread apart in a "V" peace sign.', tips: 'Fingers clearly separated in a V.', commonWords: ['Voice', 'Victory', 'Value'] },
        { letter: 'W', name: 'Letter W', category: 'alphabet', description: 'Index, middle, and ring fingers spread upwards forming a "W".', tips: 'Thumb holds pinky down against palm.', commonWords: ['Water', 'World', 'Welcome'] },
        { letter: 'X', name: 'Letter X', category: 'alphabet', description: 'Make a fist and bend the index finger into a hook like a pirate hook.', tips: 'Only index is bent/hooked; thumb rests against folded middle finger.', commonWords: ['X-ray', 'eXtra', 'eXcite'] },
        { letter: 'Y', name: 'Letter Y', category: 'alphabet', description: 'Extend thumb and pinky outwards while curling index, middle, and ring fingers into palm (shaka/hang loose sign).', tips: 'Clear silhouette of thumb and pinky extended.', commonWords: ['Yes', 'You', 'Yellow'] },
        { letter: 'Z', name: 'Letter Z', category: 'alphabet', description: 'Extend index finger and trace a "Z" in the air.', tips: 'Draw a zigzag path from left to right.', commonWords: ['Zero', 'Zone', 'Zeal'] }
    ],

    numbers: [
        { letter: '0', name: 'Number 0', category: 'numbers', description: 'Form an "O" shape with all fingertips touching the thumb.', tips: 'Circular "O" shape representing zero.', commonWords: ['None', 'Zero', 'Nil'] },
        { letter: '1', name: 'Number 1', category: 'numbers', description: 'Index finger straight up, palm facing inward/forward, other fingers closed.', tips: 'Single index finger pointing up.', commonWords: ['One', 'First', 'Single'] },
        { letter: '2', name: 'Number 2', category: 'numbers', description: 'Index and middle fingers extended in a "V" shape.', tips: 'Two fingers upright with palm forward.', commonWords: ['Two', 'Double', 'Second'] },
        { letter: '3', name: 'Number 3', category: 'numbers', description: 'Thumb, index, and middle fingers extended (in ASL, 3 uses the thumb!).', tips: 'In ASL, 3 is thumb + index + middle finger.', commonWords: ['Three', 'Triple', 'Third'] },
        { letter: '4', name: 'Number 4', category: 'numbers', description: 'Four fingers straight up, thumb tucked across palm.', tips: 'Index, middle, ring, pinky extended.', commonWords: ['Four', 'Quad', 'Fourth'] },
        { letter: '5', name: 'Number 5', category: 'numbers', description: 'All 5 fingers spread wide open with palm forward.', tips: 'Open 5 hand with fingers spread comfortably.', commonWords: ['Five', 'High Five', 'Fifth'] },
        { letter: '6', name: 'Number 6', category: 'numbers', description: 'Thumb and pinky tips touch, with index, middle, ring extended.', tips: 'Tip of pinky touches tip of thumb.', commonWords: ['Six', 'Sixth'] },
        { letter: '7', name: 'Number 7', category: 'numbers', description: 'Thumb and ring finger tips touch, other fingers extended.', tips: 'Ring finger tip touches thumb tip.', commonWords: ['Seven', 'Seventh'] },
        { letter: '8', name: 'Number 8', category: 'numbers', description: 'Thumb and middle finger tips touch, other fingers extended.', tips: 'Middle finger tip touches thumb tip.', commonWords: ['Eight', 'Eighth'] },
        { letter: '9', name: 'Number 9', category: 'numbers', description: 'Thumb and index finger tips touch (like F), other 3 fingers extended.', tips: 'Index touches thumb; 3 fingers upright.', commonWords: ['Nine', 'Ninth'] }
    ],

    commonPhrases: [
        { letter: 'ILY', name: 'I Love You', category: 'phrases', description: 'Extend thumb, index finger, and pinky simultaneously with palm forward.', tips: 'Combines letters I, L, and Y into one iconic sign.', commonWords: ['Love', 'Care', 'Affection'] },
        { letter: 'HELLO', name: 'Hello / Wave', category: 'phrases', description: 'Open flat hand near forehead or temple moving outward into a salute/wave.', tips: 'Palm faces outward, fingers together.', commonWords: ['Hi', 'Greetings', 'Welcome'] },
        { letter: 'THANK YOU', name: 'Thank You', category: 'phrases', description: 'Fingertips of flat hand touch your chin/lips, then extend forward and downward toward the person.', tips: 'Like blowing a gentle polite thank-you forward.', commonWords: ['Thanks', 'Gratitude', 'Appreciate'] },
        { letter: 'YES', name: 'Yes', category: 'phrases', description: 'Make an "S" fist and nod the fist up and down like a head nodding.', tips: 'Simulates the natural motion of a nodding head.', commonWords: ['Agree', 'Confirm', 'Sure'] },
        { letter: 'NO', name: 'No', category: 'phrases', description: 'Index and middle fingers snap down quickly onto the thumb tip like a talking bird beak.', tips: 'Quick, crisp snapping motion.', commonWords: ['Disagree', 'Decline', 'Negative'] },
        { letter: 'PLEASE', name: 'Please', category: 'phrases', description: 'Flat open palm rubs in a gentle clockwise circle over your chest / heart.', tips: 'Hand is flat against chest rotating smoothly.', commonWords: ['Request', 'Polite', 'Kindness'] },
        { letter: 'HELP', name: 'Help', category: 'phrases', description: 'Place a thumbs-up fist (A-hand) on top of your open flat non-dominant palm, and lift both together.', tips: 'The flat hand literally "supports" and lifts the fist.', commonWords: ['Assist', 'Support', 'Aid'] },
        { letter: 'SORRY', name: 'Sorry / Apology', category: 'phrases', description: 'Make an "A" fist and rub in a gentle clockwise circle over the center of your chest.', tips: 'Shows humility and warmth over the heart.', commonWords: ['Apologize', 'Regret', 'Forgive'] },
        { letter: 'GOOD', name: 'Good', category: 'phrases', description: 'Fingertips of dominant hand touch chin, then come down to rest palm-up onto non-dominant palm.', tips: 'Similar starting motion as "Thank You".', commonWords: ['Great', 'Fine', 'Well'] },
        { letter: 'BAD', name: 'Bad', category: 'phrases', description: 'Touch fingers to chin palm-in, then move hand down and flip palm outward towards the floor.', tips: 'The outward flip indicates negative/bad.', commonWords: ['Poor', 'Worse', 'Wrong'] },
        { letter: 'WATER', name: 'Water', category: 'phrases', description: 'Make a "W" hand shape and tap index finger against your chin twice.', tips: 'W-hand taps chin gently.', commonWords: ['Drink', 'Hydrate', 'Liquid'] },
        { letter: 'MORE', name: 'More', category: 'phrases', description: 'Bring fingertips of both hands together into flattened "O" shapes and tap them together twice.', tips: 'Fingertips touch in front of chest.', commonWords: ['Additional', 'Again', 'Extra'] },
        { letter: 'PEACE', name: 'Peace / Victory', category: 'phrases', description: 'Extend index and middle fingers in a "V" shape palm outward.', tips: 'Classic universal peace gesture.', commonWords: ['Harmony', 'Calm', 'Peace'] },
        { letter: 'OK', name: 'OK / All Good', category: 'phrases', description: 'Form an "O" with thumb and index, with 3 outer fingers extended upright.', tips: 'Matches the Letter F hand shape.', commonWords: ['Okay', 'Alright', 'Fine'] }
    ]
};

/**
 * Geometric heuristic detector for common phrases and dynamic gestures
 */
class CommonGestureDetector {
    /**
     * Detects special gestures from 21 MediaPipe hand landmarks
     * @param {Array<{x: number, y: number, z: number}>} landmarks
     * @returns {{detected: boolean, gesture: string|null, confidence: number}}
     */
    static detect(landmarks) {
        if (!landmarks || landmarks.length !== 21) {
            return { detected: false, gesture: null, confidence: 0 };
        }

        const isExtended = (tipIdx, pipIdx, mcpIdx) => {
            const tip = landmarks[tipIdx];
            const pip = landmarks[pipIdx];
            const mcp = landmarks[mcpIdx];
            // Y is down in screen coords, so tip.y < pip.y means pointing up
            return (tip.y < pip.y) && (pip.y < mcp.y || Math.hypot(tip.x - mcp.x, tip.y - mcp.y) > Math.hypot(pip.x - mcp.x, pip.y - mcp.y) * 1.2);
        };

        const thumbExt = Math.hypot(landmarks[4].x - landmarks[2].x, landmarks[4].y - landmarks[2].y) > 0.08;
        const indexExt = isExtended(8, 6, 5);
        const middleExt = isExtended(12, 10, 9);
        const ringExt = isExtended(16, 14, 13);
        const pinkyExt = isExtended(20, 18, 17);

        // 1. I Love You (Thumb + Index + Pinky extended, Middle & Ring folded)
        if (thumbExt && indexExt && !middleExt && !ringExt && pinkyExt) {
            return { detected: true, gesture: 'ILY', label: 'I Love You', confidence: 0.95 };
        }

        // 2. Peace / V (Index + Middle extended, Thumb, Ring, Pinky folded)
        if (!thumbExt && indexExt && middleExt && !ringExt && !pinkyExt) {
            return { detected: true, gesture: 'PEACE', label: 'Peace', confidence: 0.92 };
        }

        // 3. OK Gesture (Index and Thumb touching, Middle, Ring, Pinky extended)
        const thumbIndexDist = Math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y);
        if (thumbIndexDist < 0.05 && middleExt && ringExt && pinkyExt) {
            return { detected: true, gesture: 'OK', label: 'OK', confidence: 0.94 };
        }

        /* 
        // 4. Open Hand / Hello (All 5 fingers extended) - commented out to avoid hijacking neural net alphabet signs
        if (thumbExt && indexExt && middleExt && ringExt && pinkyExt) {
            return { detected: true, gesture: 'HELLO', label: 'Hello', confidence: 0.90 };
        }
        */

        // 5. Thumbs Up (Thumb up, other 4 fingers curled tightly)
        const thumbUp = (landmarks[4].y < landmarks[3].y) && (landmarks[3].y < landmarks[2].y);
        if (thumbUp && !indexExt && !middleExt && !ringExt && !pinkyExt) {
            return { detected: true, gesture: 'YES', label: 'Thumbs Up / Yes', confidence: 0.93 };
        }

        return { detected: false, gesture: null, confidence: 0 };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ASL_DICTIONARY, CommonGestureDetector };
} else if (typeof window !== 'undefined') {
    window.ASL_DICTIONARY = ASL_DICTIONARY;
    window.CommonGestureDetector = CommonGestureDetector;
}
