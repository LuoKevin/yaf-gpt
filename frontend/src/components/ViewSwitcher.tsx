import type { ViewMode } from "../types";

type ViewSwitcherProps = {
  activeView: ViewMode;
  isCollapsed: boolean;
  onChange: (view: ViewMode) => void;
  onToggleCollapse: () => void;
};

export function ViewSwitcher({ activeView, isCollapsed, onChange, onToggleCollapse }: ViewSwitcherProps) {
  const workspaceItems: Array<{ view: ViewMode; title: string; copy: string; icon: string }> = [
    { view: "chat", title: "Chat", copy: "Plain text", icon: "edit_note" },
    { view: "study", title: "Study", copy: "Guide builder", icon: "menu_book" },
    { view: "music", title: "Music", copy: "Prompt to track", icon: "library_music" },
    { view: "discussion", title: "Discussion", copy: "Voice and chat", icon: "forum" }
  ];

  return (
    <aside className={`sidebar ${isCollapsed ? "collapsed" : ""}`}>
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <h1 className="sidebar-brand-title">YAF-GPT</h1>
          {!isCollapsed ? <p className="sidebar-copy">Digital Sanctuary</p> : null}
        </div>
        <button
          type="button"
          className="sidebar-toggle"
          onClick={onToggleCollapse}
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <span className="material-symbols-outlined" aria-hidden="true">
            {isCollapsed ? "menu" : "menu_open"}
          </span>
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
              <span className="material-symbols-outlined">{item.icon}</span>
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
