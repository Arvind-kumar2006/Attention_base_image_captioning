import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import CaptionWords from './CaptionWords'

function setup(overrides = {}) {
  const onSelect = vi.fn()
  const words = ['a', 'dog', 'runs', 'on', 'grass']
  const utils = render(
    <CaptionWords words={words} selectedIndex={-1} onSelect={onSelect} {...overrides} />,
  )
  return { onSelect, words, ...utils }
}

describe('CaptionWords', () => {
  it('renders one button per word', () => {
    setup()
    for (const word of ['a', 'dog', 'runs', 'on', 'grass']) {
      expect(screen.getByRole('button', { name: word })).toBeTruthy()
    }
  })

  it('exposes the group with a label describing the keyboard model', () => {
    setup()
    const group = screen.getByRole('group')
    expect(group.getAttribute('aria-label')).toMatch(/arrow keys/i)
  })

  it('selects a word on click', async () => {
    const user = userEvent.setup()
    const { onSelect } = setup()
    await user.click(screen.getByRole('button', { name: 'dog' }))
    expect(onSelect).toHaveBeenCalledWith(1)
  })

  it('deselects when the selected word is clicked again', async () => {
    const user = userEvent.setup()
    const { onSelect } = setup({ selectedIndex: 1 })
    await user.click(screen.getByRole('button', { name: 'dog' }))
    expect(onSelect).toHaveBeenCalledWith(-1)
  })

  it('marks the selected word with aria-pressed', () => {
    setup({ selectedIndex: 2 })
    expect(screen.getByRole('button', { name: 'runs' }).getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByRole('button', { name: 'dog' }).getAttribute('aria-pressed')).toBe('false')
  })

  it('moves right and left with the arrow keys', async () => {
    const user = userEvent.setup()
    const { onSelect } = setup({ selectedIndex: 1 })

    screen.getByRole('button', { name: 'dog' }).focus()
    await user.keyboard('{ArrowRight}')
    expect(onSelect).toHaveBeenLastCalledWith(2)

    // focus follows selection, so stepping back returns to where we were
    await user.keyboard('{ArrowLeft}')
    expect(onSelect).toHaveBeenLastCalledWith(1)
  })

  it('moves left from the first word to index 0', async () => {
    const user = userEvent.setup()
    const { onSelect } = setup({ selectedIndex: 0 })
    screen.getByRole('button', { name: 'a' }).focus()
    await user.keyboard('{ArrowLeft}')
    expect(onSelect).toHaveBeenLastCalledWith(0)
  })

  it('jumps to the ends with Home and End', async () => {
    const user = userEvent.setup()
    const { onSelect } = setup({ selectedIndex: 2 })
    screen.getByRole('button', { name: 'runs' }).focus()

    await user.keyboard('{End}')
    expect(onSelect).toHaveBeenLastCalledWith(4)
    await user.keyboard('{Home}')
    expect(onSelect).toHaveBeenLastCalledWith(0)
  })

  it('stops at the edges rather than wrapping', async () => {
    const user = userEvent.setup()
    const { onSelect } = setup({ selectedIndex: 4 })
    screen.getByRole('button', { name: 'grass' }).focus()

    await user.keyboard('{ArrowRight}')
    expect(onSelect).toHaveBeenLastCalledWith(4)
  })

  it('clears the selection with Escape', async () => {
    const user = userEvent.setup()
    const { onSelect } = setup({ selectedIndex: 1 })
    screen.getByRole('button', { name: 'dog' }).focus()
    await user.keyboard('{Escape}')
    expect(onSelect).toHaveBeenCalledWith(-1)
  })

  it('keeps the group to a single tab stop', () => {
    setup({ selectedIndex: 2 })
    const tabbable = screen
      .getAllByRole('button')
      .filter((button) => button.getAttribute('tabindex') === '0')
    expect(tabbable).toHaveLength(1)
    expect(tabbable[0]).toHaveTextContent('runs')
  })

  it('puts the first word in the tab order when nothing is selected', () => {
    setup({ selectedIndex: -1 })
    const tabbable = screen
      .getAllByRole('button')
      .filter((button) => button.getAttribute('tabindex') === '0')
    expect(tabbable).toHaveLength(1)
    expect(tabbable[0]).toHaveTextContent('a')
  })

  it('renders nothing for an empty caption', () => {
    const { container } = render(<CaptionWords words={[]} selectedIndex={-1} onSelect={() => {}} />)
    expect(container.firstChild).toBeNull()
  })
})