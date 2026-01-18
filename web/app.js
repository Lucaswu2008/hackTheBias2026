const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('.page-section');

const lessons = [
  {
    title: 'A B C D E',
    status: 'current',
    completion: 0,
    summary: 'Focus on A-E with steady hand placement and clear finger shapes.',
    pages: [
      {
        label: 'A - Demonstration',
        title: 'A demonstration',
        body: 'Watch the closed fist shape with the thumb resting on the side.',
      },
      {
        label: 'A - Practice',
        title: 'A practice',
        body: 'Hold A steady for three seconds at center frame.',
        actionLabel: 'Open A detector',
        actionUrl: 'http://localhost:5000/detect?letter=A',
      },
      {
        label: 'B - Demonstration',
        title: 'B demonstration',
        body: 'Extend fingers upward with the thumb across the palm.',
      },
      {
        label: 'B - Practice',
        title: 'B practice',
        body: 'Keep fingers together and check palm facing the camera.',
        actionLabel: 'Open B detector',
        actionUrl: 'http://localhost:5000/detect?letter=B',
      },
      {
        label: 'C - Demonstration',
        title: 'C demonstration',
        body: 'Curve the hand to form a clear C shape.',
      },
      {
        label: 'C - Practice',
        title: 'C practice',
        body: 'Hold C steady with even curve and spacing.',
        actionLabel: 'Open C detector',
        actionUrl: 'http://localhost:5000/detect?letter=C',
      },
      {
        label: 'D - Demonstration',
        title: 'D demonstration',
        body: 'Raise index finger with thumb touching middle finger.',
      },
      {
        label: 'D - Practice',
        title: 'D practice',
        body: 'Keep the circle tight and the index finger vertical.',
        actionLabel: 'Open D detector',
        actionUrl: 'http://localhost:5000/detect?letter=D',
      },
      {
        label: 'E - Demonstration',
        title: 'E demonstration',
        body: 'Curl fingers inward with the thumb tucked.',
      },
      {
        label: 'E - Practice',
        title: 'E practice',
        body: 'Hold the compact shape steady and centered.',
        actionLabel: 'Open E detector',
        actionUrl: 'http://localhost:5000/detect?letter=E',
      },
    ],
  },
  {
    title: 'F G H I',
    status: 'upcoming',
    completion: 0,
    summary: 'Work through F-I with balanced spacing and steady pacing.',
  },
  {
    title: 'J K L M N O',
    status: 'upcoming',
    completion: 0,
    summary: 'Practice J-O with clean hand orientation and deliberate motion.',
  },
  {
    title: 'P Q R S T',
    status: 'upcoming',
    completion: 0,
    summary: 'Lock in P-T with sharp, readable shapes and pauses.',
  },
  {
    title: 'U V W X Y Z',
    status: 'upcoming',
    completion: 0,
    summary: 'Finish the alphabet with consistent spacing and wrist alignment.',
  },
  {
    title: 'Warmup loop',
    status: 'upcoming',
    completion: 0,
    summary: 'A quick warmup to prep hands before drilling.',
  },
  {
    title: 'Speed cadence',
    status: 'upcoming',
    completion: 0,
    summary: 'Practice short bursts to build speed without losing clarity.',
  },
  {
    title: 'Accuracy check',
    status: 'upcoming',
    completion: 0,
    summary: 'Revisit tricky letters and focus on confidence.',
  },
  {
    title: 'Confidence pass',
    status: 'upcoming',
    completion: 0,
    summary: 'One full run with pauses to confirm each shape.',
  },
  {
    title: 'Endurance round',
    status: 'upcoming',
    completion: 0,
    summary: 'Sustain steady output for a longer session.',
  },
  {
    title: 'Review recap',
    status: 'upcoming',
    completion: 0,
    summary: 'Summarize key improvements and goals for next time.',
  },
];

const lessonList = document.getElementById('lessonList');
const lessonTitle = document.getElementById('lessonTitle');
const lessonSummary = document.getElementById('lessonSummary');
const lessonCompletion = document.getElementById('lessonCompletion');
const lessonStatus = document.getElementById('lessonStatus');
const lessonAction = document.getElementById('lessonAction');
const lessonDeepPreview = document.getElementById('lessonDeepPreview');
const lessonPlayerTitle = document.getElementById('lessonPlayerTitle');
const lessonExit = document.getElementById('lessonExit');
const lessonPageLabel = document.getElementById('lessonPageLabel');
const lessonPageTitle = document.getElementById('lessonPageTitle');
const lessonPageBody = document.getElementById('lessonPageBody');
const lessonDots = document.getElementById('lessonDots');
const lessonBack = document.getElementById('lessonBack');
const lessonNext = document.getElementById('lessonNext');
const lessonDetectorWrap = document.getElementById('lessonDetectorWrap');
const lessonDetectorFrame = document.getElementById('lessonDetectorFrame');
const lessonDetectorPlaceholder = document.getElementById('lessonDetectorPlaceholder');
const homeContinue = document.getElementById('homeContinue');
const homeGoToLesson = document.getElementById('homeGoToLesson');

const lessonElements = [];
let activeLessonIndex = 0;
let activePageIndex = 0;
let activePages = [];
let activePracticeLetter = null;
let unlockedPages = new Set();

const getCurrentLessonIndex = () => {
  const currentIndex = lessons.findIndex((lesson) => lesson.status === 'current');
  return currentIndex >= 0 ? currentIndex : 0;
};

const getLessonPages = (lesson) => {
  if (lesson.pages && lesson.pages.length > 0) {
    return lesson.pages;
  }
  return [
    {
      title: 'Lesson overview',
      body: lesson.summary,
    },
  ];
};

const getPracticeLetter = (page) => {
  if (!page.actionUrl) return null;
  try {
    const url = new URL(page.actionUrl);
    return url.searchParams.get('letter');
  } catch {
    return null;
  }
};

const statusLabel = (status) => {
  if (status === 'current') return 'Current';
  return 'Upcoming';
};

const primaryActionLabel = (lesson) => {
  if (lesson.status === 'current') return 'Start lesson';
  return 'Preview';
};

const updateLessonDetail = (lesson, index) => {
  if (!lessonTitle) return;
  const lessonNumber = index + 1;
  lessonTitle.textContent = `Lesson ${lessonNumber}: ${lesson.title}`;
  lessonSummary.textContent = lesson.summary;
  if (lessonDeepPreview) {
    lessonDeepPreview.textContent = lesson.detail || lesson.summary;
  }
  lessonCompletion.textContent = `${lesson.completion}%`;
  lessonStatus.textContent = statusLabel(lesson.status);
  lessonAction.textContent = primaryActionLabel(lesson);
};

const renderLessonDots = () => {
  if (!lessonDots) return;
  lessonDots.innerHTML = '';
  activePages.forEach((_, index) => {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'lesson-dot';
    dot.setAttribute('aria-label', `Lesson page ${index + 1}`);
    dot.addEventListener('click', () => {
      activePageIndex = index;
      renderLessonPage();
    });
    lessonDots.appendChild(dot);
  });
};

const renderLessonPage = () => {
  const page = activePages[activePageIndex];
  if (!page) return;
  activePracticeLetter = getPracticeLetter(page);
  if (lessonPageLabel) {
    lessonPageLabel.textContent =
      page.label || `Lesson page ${activePageIndex + 1} of ${activePages.length}`;
  }
  if (lessonPageTitle) {
    lessonPageTitle.textContent = page.title;
  }
  if (lessonPageBody) {
    lessonPageBody.textContent = page.body;
  }
  if (lessonDetectorWrap && lessonDetectorFrame && lessonDetectorPlaceholder) {
    if (page.actionUrl) {
      lessonDetectorFrame.src = page.actionUrl;
      lessonDetectorPlaceholder.style.display = 'none';
    } else {
      lessonDetectorFrame.src = 'about:blank';
      lessonDetectorPlaceholder.style.display = 'grid';
    }
  }
  if (lessonDots) {
    Array.from(lessonDots.children).forEach((dot, index) => {
      dot.classList.toggle('is-active', index === activePageIndex);
    });
  }
  if (lessonBack) {
    lessonBack.disabled = activePageIndex === 0;
  }
  if (lessonNext) {
    const isLastPage = activePageIndex === activePages.length - 1;
    const isUnlocked = unlockedPages.has(activePageIndex);
    if (Boolean(activePracticeLetter)) {
      lessonNext.disabled = isLastPage || !isUnlocked;
    } else {
      lessonNext.disabled = isLastPage;
    }
  }
};

const loadLessonPages = (lesson) => {
  activePages = getLessonPages(lesson);
  activePageIndex = 0;
  unlockedPages = new Set();
  renderLessonDots();
  renderLessonPage();
};

const selectLesson = (index, { scrollIntoView } = { scrollIntoView: false }) => {
  const lesson = lessons[index];
  if (!lesson) return;
  activeLessonIndex = index;
  lessonElements.forEach((item, idx) => {
    item.classList.toggle('is-active', idx === index);
    item.setAttribute('aria-selected', idx === index ? 'true' : 'false');
  });
  updateLessonDetail(lesson, index);
  loadLessonPages(lesson);
  if (scrollIntoView) {
    lessonElements[index].scrollIntoView({ block: 'nearest' });
  }
};

const renderLessons = () => {
  if (!lessonList) return;
  lessonList.innerHTML = '';
  lessonElements.length = 0;
  lessons.forEach((lesson, index) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'lesson-item';
    item.innerHTML = `
      <p class="lesson-title">Lesson ${index + 1}: ${lesson.title}</p>
      <p class="lesson-meta">Status: ${statusLabel(lesson.status)}</p>
    `;
    item.addEventListener('click', () => selectLesson(index));
    lessonList.appendChild(item);
    lessonElements.push(item);
  });
  selectLesson(getCurrentLessonIndex(), { scrollIntoView: true });
};

const showLessonPlayer = (lesson, index) => {
  const lessonNumber = index + 1;
  lessonPlayerTitle.textContent = `Lesson ${lessonNumber}: ${lesson.title}`;
  loadLessonPages(lesson);
  showSection('lesson');
  history.replaceState(null, '', '#lesson');
};

const hideLessonPlayer = () => {
  showSection('learning');
  history.replaceState(null, '', '#learning');
};

const showSection = (targetId) => {
  sections.forEach((section) => {
    section.classList.toggle('is-visible', section.id === targetId);
  });

  navLinks.forEach((link) => {
    link.classList.toggle('is-active', link.dataset.target === targetId);
  });

  document.body.classList.toggle('no-scroll', targetId === 'learning' || targetId === 'lesson');

  if (targetId === 'learning') {
    selectLesson(getCurrentLessonIndex(), { scrollIntoView: true });
  }
};

navLinks.forEach((link) => {
  link.addEventListener('click', () => {
    const target = link.dataset.target;
    if (!target) return;
    showSection(target);
    history.replaceState(null, '', `#${target}`);
  });
});

if (lessonAction) {
  lessonAction.addEventListener('click', () => {
    const lesson = lessons[activeLessonIndex];
    if (!lesson) return;
    showLessonPlayer(lesson, activeLessonIndex);
  });
}

if (lessonExit) {
  lessonExit.addEventListener('click', () => {
    hideLessonPlayer();
  });
}

if (lessonBack) {
  lessonBack.addEventListener('click', () => {
    if (activePageIndex > 0) {
      activePageIndex -= 1;
      renderLessonPage();
    }
  });
}

if (lessonNext) {
  lessonNext.addEventListener('click', () => {
    if (activePageIndex < activePages.length - 1) {
      activePageIndex += 1;
      renderLessonPage();
    }
  });
}

window.addEventListener('message', (event) => {
  if (event.origin !== 'http://localhost:5000') return;
  const data = event.data;
  if (!data || data.type !== 'detector-status') return;
  if (!activePracticeLetter) return;
  if ((data.letter || '').toUpperCase() !== activePracticeLetter.toUpperCase()) return;
  if (lessonNext) {
    const isLastPage = activePageIndex === activePages.length - 1;
    if (data.targetMet) {
      unlockedPages.add(activePageIndex);
    }
    const isUnlocked = unlockedPages.has(activePageIndex);
    lessonNext.disabled = isLastPage || !isUnlocked;
  }
});

if (homeContinue) {
  homeContinue.addEventListener('click', () => {
    const currentIndex = getCurrentLessonIndex();
    selectLesson(currentIndex);
    showLessonPlayer(lessons[currentIndex], currentIndex);
  });
}

if (homeGoToLesson) {
  homeGoToLesson.addEventListener('click', () => {
    showSection('learning');
    history.replaceState(null, '', '#learning');
  });
}

const initialTarget = window.location.hash.replace('#', '') || 'home';
renderLessons();
showSection(initialTarget);
