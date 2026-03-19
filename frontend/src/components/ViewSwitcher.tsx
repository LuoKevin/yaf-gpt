import type { ViewMode } from "../types";

type ViewSwitcherProps = {
  activeView: ViewMode;
  isCollapsed: boolean;
  onChange: (view: ViewMode) => void;
  onToggleCollapse: () => void;
};

export function ViewSwitcher({ activeView, isCollapsed, onChange, onToggleCollapse }: ViewSwitcherProps) {
  const workspaceItems: Array<{ view: ViewMode; title: string; copy: string; icon: JSX.Element }> = [
    {
      view: "chat",
      title: "Chat",
      copy: "Plain text",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4 19.5V5.75A1.75 1.75 0 0 1 5.75 4h7.5" />
          <path d="M14 4h4.25A1.75 1.75 0 0 1 20 5.75V18.25A1.75 1.75 0 0 1 18.25 20H8" />
          <path d="M8 16l7-7" />
          <path d="M13.5 8.5H16" />
          <path d="M8 20h8" />
        </svg>
      )
    },
    {
      view: "study",
      title: "Study",
      copy: "Guide builder",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4.75 5.5A2.75 2.75 0 0 1 7.5 2.75H19.25V18.5H7.5A2.75 2.75 0 0 0 4.75 21.25Z" />
          <path d="M7.5 2.75V21.25" />
          <path d="M9.75 7.5H15.5" />
          <path d="M9.75 11H15.5" />
        </svg>
      )
    },
    {
      view: "music",
      title: "Music",
      copy: "Prompt to track",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14.5 4.75v10.5" />
          <path d="M14.5 6.25 19 5v9.5" />
          <path d="M14.5 15.25a2.75 2.75 0 1 1-2.75-2.75 2.75 2.75 0 0 1 2.75 2.75Z" />
          <path d="M19 16.75A2.75 2.75 0 1 1 16.25 14 2.75 2.75 0 0 1 19 16.75Z" />
        </svg>
      )
    },
    {
      view: "discussion",
      title: "Discussion",
      copy: "Voice and chat",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M5.75 6.75A2.75 2.75 0 0 1 8.5 4h7A2.75 2.75 0 0 1 18.25 6.75v4.5A2.75 2.75 0 0 1 15.5 14h-4l-3.75 3v-3A2.75 2.75 0 0 1 5.75 11.25Z" />
          <path d="M9.25 8.75h5.5" />
          <path d="M9.25 11.25h3.75" />
        </svg>
      )
    }
  ];

  return (
    <aside className={`sidebar ${isCollapsed ? "collapsed" : ""}`}>
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <p className="eyebrow">yaf-gpt</p>
          {!isCollapsed ? <p className="sidebar-copy">Workspaces</p> : null}
        </div>
        <button
          type="button"
          className="sidebar-toggle"
          onClick={onToggleCollapse}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            {isCollapsed ? (
              <path d="m9 6 6 6-6 6" />
            ) : (
              <path d="m15 6-6 6 6 6" />
            )}
          </svg>
        </button>
      </div>

      <nav className="view-switcher" aria-label="Workspace selection">
        {workspaceItems.map((item) => (
          <button
            key={item.view}
            type="button"
            className={`view-button ${activeView === item.view ? "active" : ""}`}
            onClick={() => onChange(item.view)}
            title={isCollapsed ? item.title : undefined}
            aria-label={item.title}
          >
            <span className="view-icon" aria-hidden="true">
              {item.icon}
            </span>
            {!isCollapsed ? (
              <span className="view-copy-block">
                <span className="view-title">{item.title}</span>
                <span className="view-copy">{item.copy}</span>
              </span>
            ) : null}
          </button>
        ))}
      </nav>
    </aside>
  );
}
